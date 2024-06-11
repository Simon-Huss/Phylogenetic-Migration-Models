# Exponential distribution
function exp_dist(rate)
    return rand(Distributions.Exponential(1/rate))
end 

# Transforms a Q-matrix into a corresponding jump matrix J as well as the holding time for the state
function Q_to_J(Q, state)
    J = Q./(-Q[state])
    J[state] = 0
    return [J,  -Q[state]]
end

# For integer states on sparse vectors
function simulate_CTMC(D, func::Function, state::Int, T; kwargs... )
    transition_vector, holding_time = func(D, state; kwargs... )
    t = exp_dist(holding_time)
    counter = 0
    X = state
    while t < T
        counter += 1
        X = sample(transition_vector.nzind, Weights(transition_vector.nzval))
        transition_vector, holding_time = func(D, X; kwargs...)
        t += exp_dist(holding_time)
    end 
    if haskey(kwargs, :counter)
        kwargs[:counter].value += counter
    end
    return X
end

# For general vector states
function simulate_CTMC(D, func::Function, state, T; kwargs... )
    X = sample(1:length(state), Weights(state))
    transition_vector, holding_time = func(D, X; kwargs... )   
    t = exp_dist(holding_time)
    counter = 0 
    while t < T 
        counter += 1
        X = sample(1:length(state), Weights(transition_vector))
        transition_vector, holding_time = func(D, X; kwargs...)
        t += exp_dist(holding_time)
    end 
    if haskey(kwargs, :counter)
        kwargs[:counter].value += counter
    end
    X_final = zeros(size(state))
    X_final[X] = 1
    return X_final
end

# Get the 3D coordinates of the cities from the dataset 
function get_coords(n=1000; kwargs...)
    path = joinpath(@__DIR__, "../Data/") 
   
    cd(path) do
        global data_ref = CSV.read("geonames_data_10_000.csv", DataFrame);
    end
    data = copy(data_ref);
    
    sort!(data, [:Population], rev=[true])
    cords = data.Coordinates[1:n]
    x = Float64[]
    y = Float64[]
    z = Float64[]

    # Extract coordinates and populate vectors
    for coord_str in cords
        lat_str, lon_str = split(coord_str, ",")
        phi = parse(Float64, lon_str);
        theta = parse(Float64, lat_str);
        push!(x, cosd(theta)*cosd(phi))
        push!(y, cosd(theta)*sind(phi))
        push!(z, sind(theta))
        
    end
    
    M = [x y z]'

    if haskey(kwargs, :return_names) && kwargs[:return_names]
        if haskey(kwargs, :return_countries) && kwargs[:return_countries]
            return M, data.Name[1:n], data."Country name EN"[1:n]
        else
            return M, data.Name[1:n]
        end    
    else
        return  M
    end
end

# Get the 2D coordinates of the cities from the dataset 
function get_coords_2D(n=1000; kwargs...)
    path = joinpath(@__DIR__, "../Data/") 

    cd(path) do
        global data_ref = CSV.read("geonames_data_10_000.csv", DataFrame);
    end
    data = copy(data_ref);    

    sort!(data, [:Population], rev=[true])
    cords = data.Coordinates[1:n]
    x = Float64[]
    y = Float64[]

    # Extract coordinates and populate vectors
    for coord_str in cords
        lat_str, lon_str = split(coord_str, ",")
        push!(x, parse(Float64, lon_str))
        push!(y, parse(Float64, lat_str))
    end
    
    M = [x y]'

    if haskey(kwargs, :return_names) && kwargs[:return_names]
        return M, data.Name[1:n]
    else
        return  M
    end
end

# Function to simulate nearby migration using a KNN as well as non-local flight migration between the "flights" largest cities if use_flights is true
function KNN_migration(D, X, prop_const_alpha, flights = 10, get_Q=false, use_flights=false, phi = 0.25)
    
    # Function to inversely scale the migration according to distances with proportionality constant alpha
    inv_avoid_zero(x) = x == 0 ? 0 : prop_const_alpha/x
    Q_row = spzeros(length(D[1]))
    Q_row[D[1][X]] = inv_avoid_zero.(D[2][X]) # D is a KNN tuple with the first element being the indices and the second element being the distances
    
    # If the city is a flight city, the migration is distributed to the other flight cities as well with a proportionality constant phi, it is assumed that the "flights" first cities are the cities with flights
    if X <= flights && use_flights
        Q_row[1:flights] .+= phi * sum(Q_row) / (flights-1) # Also incorrectly adding migration to the city itself, but it's corrected below 
    end

    Q_row[X] = 0 # As the incorrectly assigned Q_row[X] value should not be included in the sum below it is set to zero as the value should be the negative sum of the other elements
    Q_row[X] = -sum(Q_row) # Setting the current state value such that the sum of the row is zero

    if get_Q
        return Q_row
    else
        return Q_to_J(Q_row, X)
    end
end

function distance_matrix(coords, use_flights = true, flight_cities = 50, gamma = 10)
    # Great circle geodesic distance, using the fact that cos(x) = 1 - CosineDist(a, b) and we want the distance in radians i.e the angle between them (as we have normed coordinate vectors)
    D = acos.(1.0.-pairwise(CosineDist(), coords))
    
    # Scaling factor gamma to make the distances respect flights shortening the distances
    if use_flights
        D[1:min(length(D[1, :]), flight_cities), 1:min(length(D[1, :]), flight_cities)] ./= gamma
    end
    return D
end

# Performs classical Multidimensional Scaling on the given coordinates
function MDS_embedding(coords, embedding_dim, use_flights, flight_cities = 0)
    D = distance_matrix(coords, use_flights, flight_cities)
    embedding = predict(fit(MDS, D; maxoutdim=embedding_dim, distances=true))
    return embedding
end

# Performs Landmark Multidimensional Scaling on the given coordinates
function Landmark_MDS(coords, embedding_dim, use_flights, flight_cities, initial_mds_datapoints; eig_size_limit = 1e-6)

    D = distance_matrix(coords[:, 1:initial_mds_datapoints], use_flights, flight_cities)
    M = fit(MDS, D; maxoutdim=embedding_dim, distances=true)
    number_of_cities = length(coords[1, :])

    delta = D.^2 # with columns delta_i = squared distances from landmark i to all other landmarks
    eigs = eigvals(M)
    vecs = eigvecs(M)
    println("Landmark eigs: ", eigs)

    # Counts the number of eigenvalues above the size limit and adjusts the embedding dimension accordingly as too small eigenvalues can cause stability problems
    non_zero_eigs = count(x -> x > eig_size_limit, eigs)
    if length(eigs) > non_zero_eigs
        @warn "Eigenvalues below the size limit: $eig_size_limit detected, only using $non_zero_eigs eigenvalues and thus lowering the embedding dimension from $embedding_dim to $non_zero_eigs."
    end
    embedding_dim = min(embedding_dim, non_zero_eigs)
    embedding = zeros(embedding_dim, number_of_cities)

    # Re-initializing the MDS model with the new adjusted embedding dimension
    M = fit(MDS, D; maxoutdim=embedding_dim, distances=true)
    embedding[:, 1:initial_mds_datapoints] = predict(M)

    eigs = eigs[1:non_zero_eigs]
    vecs = vecs[:, 1:non_zero_eigs]
    Lkpt = transpose(vecs) ./ sqrt.(eigs) # Pseudo inverse tranpose of Lk
    delta_mu = mean(delta, dims=2)

    # Triangulation
    # Input: delta_a = squared distance from point a to all landmark points
    # Output: x_a = embedded point triangulated according to formula -1/2 * Lkpt * (delta_a - delta_mu)
    
    for i=initial_mds_datapoints+1:number_of_cities
        dists = coords[:, i] .- coords[:, 1:initial_mds_datapoints]
        delta_a = norm.(eachcol(dists)).^2
        x_a = -1/2 * Lkpt * (delta_a - delta_mu)
        embedding[:, i] = x_a
    end

    return embedding
end

# Optimizes the diffusion rates for the continuous BrownianMotion BranchModel given the tree with the leaf node data on it
function optimize_diffusions(tree, dim, gss_lower_bound, gss_upper_bound, error_tolerance = 1e-10; return_lls = false)
    lls = []
    diffusions = []
    model_functions(x) = [BrownianMotion(0.0, x) for d in 1:dim]
    for i = 1:dim
        ll(x) = log_likelihood!(tree, model_functions(x), partition_list=[i])
        push!(lls, ll)
        push!(diffusions, golden_section_maximize(ll, gss_lower_bound, gss_upper_bound, identity, error_tolerance))
        if  abs(gss_upper_bound - diffusions[i]) < 3 * error_tolerance # Arbitrary scaling factor of 3
            @warn "Optimal diffusion, $(diffusions[i]), is close to the upper bound for GSS search, consider increasing the upper bound."
        end
    end
    if return_lls
        return diffusions, lls
    else
        return diffusions
    end
end

# Optimizes the r-scaling parameter for the PiQ BranchModel given the tree with the leaf node data on it
function optimize_rscaling(PiQ_model, tree, gss_lower_bound, gss_upper_bound, error_tolerance = 1e-5)
    pi = eq_freq(PiQ_model)
    ll(x) = log_likelihood!(tree, PiQ(x, pi))
    optimal_r = golden_section_maximize(ll, gss_lower_bound, gss_upper_bound, identity, error_tolerance)
    if  abs(gss_upper_bound - optimal_r) < 3 * error_tolerance # Arbitrary scaling factor of 3
        @warn "Optimal r scaling, $optimal_r, is close to the upper bound for GSS search, consider increasing the upper bound."
    end
    if (optimal_r < 10 * error_tolerance)
        @warn "Optimal r scaling, $optimal_r, is close to the error tolerance, $error_tolerance, consider decreasing the tolerance for a more accurate result."
    end
    optimal_model = PiQ(optimal_r, pi)
    return optimal_model, optimal_r
end

# Initializes the tree with a uniform distribution for the given partition
function uniform_init!(tree, n, partition) 
    partitions = []
    if typeof(partition) == CustomDiscretePartition
        X_0 = ones(n)/n
        partitions = CustomDiscretePartition(n, 1)
        partitions.state[:, 1] = X_0 
    elseif typeof(partition) == GaussianPartition
        initial_mean = 0
        initial_var = 1000
        partitions = [GaussianPartition(initial_mean, initial_var) for _ in 1:n]
    elseif typeof(partition) == CustomDiscretePartitionFloat32
        X_0 = ones(n)/n
        partitions = CustomDiscretePartitionFloat32(n, 1)
        partitions.state[:, 1] = X_0
    elseif typeof(partition) == CustomDiscretePartitionFloat16
        X_0 = ones(n)/n
        partitions = CustomDiscretePartitionFloat16(n, 1)
        partitions.state[:, 1] = X_0
    else
        error("Error - The specified partition has not been configired.")
    end
    internal_message_init!(tree, partitions)
end

# Initialize the node_data dictionaries to store the migration and simulation outcomes
function initialize_node_data(tree)
    for node in getleaflist(tree)
        node.node_data = Dict()
    end
    for node in getnonleaflist(tree)
        node.node_data = Dict()
    end
end

# Calculates the migrations from the root to the leaf nodes
function root2tip_migrations(tree)
    migrations_list = []
    mapping = Dict()
    for (index, node) in enumerate(getleaflist(tree))
        mapping[node] = index
        migrations = node.node_data["migrations"]
        while !isnothing(node.parent)
            node = node.parent
            migrations += node.node_data["migrations"]
        end
        push!(migrations_list, migrations)
    end
    return migrations_list, mapping
end
