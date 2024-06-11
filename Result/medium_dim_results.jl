include("../utils/load.jl")

iterations = 1
total_time = @elapsed begin

number_of_cities = 10_000 # Number of cities i.e dimensionality of the data
number_of_neighbours_knn = 200 # Number of neighbours in the KNN algorithm

use_flights = false # Use flights in the distance matrix D and transition matrix Q if true, else use only geodesic distance and local migration
flight_cities = 400 # Number of cities with flights

embedding_dims = [25 50 75]
initial_mds_datapoints = 800 # Number of dimensions in the initial MDS embedding for the landmark algorithm
prop_const_alpha = 2.5*1e-7 


# ---  GSS search parameters ---
gss_lower_bound = 0
gss_upper_bound_rscaling = 2
gss_upper_bound_diffusion = 200
error_tolerance = 1e-6

coordinates, cities, countries = get_coords(number_of_cities, return_names=true, return_countries=true)

PiQ_prob_of_start_indices = []
mds_prob_of_start_indices = [[] for _ in embedding_dims]
landmark_prob_of_start_indices = [[] for _ in embedding_dims]

minimum_migrations = []
average_migrations = []
total_branchlengths = []
total_migrations = []

# --- KNN ---
knn_tree = KDTree(coordinates)
K = knn(knn_tree, coordinates, number_of_neighbours_knn+1)


# --- MDS --- 
embeddings_MDS = []
embeddings_landmark = []
for embedding_dim in embedding_dims
    push!(embeddings_MDS, MDS_embedding(coordinates, embedding_dim, use_flights, flight_cities))
    push!(embeddings_landmark, Landmark_MDS(coordinates, embedding_dim, use_flights, flight_cities, initial_mds_datapoints))
end


for i = 1:iterations

    println((i-1)/iterations*100, "%")
    
    start_index = rand(1:number_of_cities)

    # --- Tree ---
    println("Creating tree ", now())
    local tree = sim_tree(100_000, 1000.0, 0.05)
    initialize_node_data(tree)
    
    #----- Setting up model for simulation -------
    println("Sampling down ", now())
    local partition = Partition_CTMC(start_index)
    internal_message_init!(tree, partition)
    local model = Gillespie_CTMC(K, (args...; kwargs...) -> KNN_migration(args..., prop_const_alpha, flight_cities, false, use_flights))
    # Sampling the states for the tree nodes
    sample_down!(tree, model)
    save_actual_states!(tree, cities)

    CustomDiscretePartition
    
    println("Starting PiQ ", now())
    # --- PiQ with ones inference ---
    uniform_init!(tree, number_of_cities, CustomDiscretePartitionFloat32(number_of_cities, 1))
    retrieve_states_discrete!(tree)
    ones_model = PiQ(ones(number_of_cities))
    println("Starting r-scaling ", now())
    PiQ_model, opt_r = optimize_rscaling(ones_model, tree, gss_lower_bound, gss_upper_bound_rscaling)
    println("PiQ inference ", now())
    PiQ_model_dictionary = marginal_state_dict(tree, PiQ_model)

    
    println("Starting MDS model ", now())
    # --- continuous embedding MDS inference ---
    marginal_state_dictionaries_MDS = []
    for (index, embedding_MDS) in enumerate(embeddings_MDS)
        uniform_init!(tree, embedding_dims[index], GaussianPartition())
        retrieve_states_continuous!(tree, embedding_MDS)
        diffusions = optimize_diffusions(tree, embedding_dims[index], gss_lower_bound, gss_upper_bound_diffusion, error_tolerance)
        models = [BrownianMotion(0.0, diffusions[i]) for i in 1:embedding_dims[index]]
        push!(marginal_state_dictionaries_MDS, marginal_state_dict(tree, models))
    end

    println("Starting landmark model ", now())
    # --- continuous embedding landmark MDS inference ---
    marginal_state_dictionaries_landmark = []
    for (index, embedding_landmark) in enumerate(embeddings_landmark)
        uniform_init!(tree, embedding_dims[index], GaussianPartition())
        retrieve_states_continuous!(tree, embedding_landmark)
        diffusions = optimize_diffusions(tree, embedding_dims[index], gss_lower_bound, gss_upper_bound_diffusion, error_tolerance)
        models = [BrownianMotion(0.0, diffusions[i]) for i in 1:embedding_dims[index]]
        push!(marginal_state_dictionaries_landmark, marginal_state_dict(tree, models))
    end

    # --- Accuracy measures ---
    println("Starting accuracy measurements ", now())

    PiQ_root_probs = get_probabilities(PiQ_model_dictionary[tree])

    mds_root_probs = []
    landmark_root_probs = []

    for index in 1:length(embedding_dims)
        push!(mds_root_probs, get_probabilities(marginal_state_dictionaries_MDS[index][tree], embeddings_MDS[index]))
        push!(landmark_root_probs, get_probabilities(marginal_state_dictionaries_landmark[index][tree], embeddings_landmark[index]))
    end

    PiQ_prob_of_start_index = PiQ_root_probs[start_index]
    push!(PiQ_prob_of_start_indices, PiQ_prob_of_start_index)
    
    
    for index in 1:length(embedding_dims)
        push!(mds_prob_of_start_indices[index], mds_root_probs[index][start_index])
        push!(landmark_prob_of_start_indices[index], landmark_root_probs[index][start_index])
    end
    
    # --- Tree statistics ---
    migrations = [x.node_data["migrations_from_root"] for x in getleaflist(tree)]
    bl = branchlengths(tree)
    tm = model.counter.value
    push!(total_migrations, tm)
    push!(total_branchlengths, bl)
    push!(average_migrations, tm/bl)
    push!(minimum_migrations, minimum(migrations))
   
end # for loop
end # total time

println("100.0%")

println("Total time: ", total_time, " and time per iteration: ", total_time/iterations)

data = DataFrame(
    
    PIQ_ROOT_PROB_OF_START_INDICES = PiQ_prob_of_start_indices,
    MDS_ROOT_PROB_OF_START_INDICES = format_result(mds_prob_of_start_indices),
    LANDMARK_ROOT_PROB_OF_START_INDICES = format_result(landmark_prob_of_start_indices),
    
    MINIMUM_MIGRATIONS = minimum_migrations,
    TOTAL_MIGRATIONS = total_migrations,
    AVERAGE_MIGRATIONS = average_migrations,
    TOTAL_BRANCHLENGTHS = total_branchlengths,
    DIMS = [embedding_dims for _ = 1:iterations]
)

# Write to file
cd(@__DIR__)

prefix = "medium_dim_results_"

if use_flights 
    global prefix *= "flights_"
end

filename = string(prefix, Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), ".csv")
CSV.write(filename, data)
