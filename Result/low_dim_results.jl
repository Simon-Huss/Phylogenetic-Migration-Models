include("../utils/load.jl")

iterations = 1
total_time = @elapsed begin

number_of_cities = 250 # Number of cities i.e dimensionality of the data
number_of_neighbours_knn = 5 # Number of neighbours in the KNN algorithm

use_flights = true # Use flights in the distance matrix D and transition matrix Q if true, else use only geodesic distance and local migration
flight_cities = 10 # Number of cities with flights

embedding_dims = [3, 4, 5, 6, 7] # Embedding dimension i.e number of dimensions in our embedded vector space
initial_mds_datapoints = 20 # Number of dimensions in the initial MDS embedding for the landmark algorithm

prop_const_alpha = 1e-5 # Transition rate proportionality constant

# ---  GSS search parameters ---
gss_lower_bound = 0
gss_upper_bound_rscaling = 2
gss_upper_bound_diffusion = 200
error_tolerance = 1e-6

coordinates, cities, countries = get_coords(number_of_cities, return_names=true, return_countries=true)

PiQ_kl_divergences_forward = []
mds_kl_divergences_forward = [[] for _ in embedding_dims]
landmark_kl_divergences_forward = [[] for _ in embedding_dims]

PiQ_kl_divergences_backward = []
mds_kl_divergences_backward = [[] for _ in embedding_dims]
landmark_kl_divergences_backward = [[] for _ in embedding_dims]

real_q_prob_of_start_indices = []
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

Q = zeros(number_of_cities, number_of_cities);
for i=1:number_of_cities
    Q[i,:] = KNN_migration(K, i, prop_const_alpha, flight_cities, true, use_flights) 
end

# --- MDS --- 
embeddings_MDS = []
embeddings_landmark = []

for embedding_dim in embedding_dims
    push!(embeddings_MDS, MDS_embedding(coordinates, embedding_dim, use_flights, flight_cities))
    push!(embeddings_landmark, Landmark_MDS(coordinates, embedding_dim, use_flights, use_flights, initial_mds_datapoints)) 
end


for i = 1:iterations

    println((i-1)/iterations*100, "%")
    
    start_index = rand(1:number_of_cities)

    # --- Tree ---
    tree = sim_tree(2500, 1000.0, 0.05)
    initialize_node_data(tree)
    

    #--- Setting up model for simulation ---
    partition = Partition_CTMC(start_index)
    internal_message_init!(tree, partition)
    model = Gillespie_CTMC(K, (args...; kwargs...) -> KNN_migration(args..., prop_const_alpha, flight_cities, false, use_flights))
    
    # --- Sampling the states for the tree nodes ---
    sample_down!(tree, model)
    save_actual_states!(tree, cities)


    # --- PiQ with ones inference ---
    uniform_init!(tree, number_of_cities, CustomDiscretePartition(number_of_cities, 1))
    retrieve_states_discrete!(tree)
    ones_model = PiQ(ones(number_of_cities))
    PiQ_model, opt_r = optimize_rscaling(ones_model, tree, gss_lower_bound, gss_upper_bound_rscaling)
    PiQ_model_dictionary = marginal_state_dict(tree, PiQ_model)

    # --- Real Q inference ---
    uniform_init!(tree, number_of_cities, CustomDiscretePartition(number_of_cities, 1))
    retrieve_states_discrete!(tree)
    real_Q_model = GeneralCTMC(Q)
    real_Q_dictionary = marginal_state_dict(tree, real_Q_model)
    
    # --- Continuous embedding MDS inference ---
    marginal_state_dictionaries_MDS = []
    for (index, embedding_MDS) in enumerate(embeddings_MDS)
        uniform_init!(tree, embedding_dims[index], GaussianPartition())
        retrieve_states_continuous!(tree, embedding_MDS)
        diffusions = optimize_diffusions(tree, embedding_dims[index], gss_lower_bound, gss_upper_bound_diffusion, error_tolerance)
        models = [BrownianMotion(0.0, diffusions[i]) for i in 1:embedding_dims[index]]
        push!(marginal_state_dictionaries_MDS, marginal_state_dict(tree, models))
    end

    # --- Continuous embedding landmark MDS inference  ---
    marginal_state_dictionaries_landmark = []
    for (index, embedding_landmark) in enumerate(embeddings_landmark)
        uniform_init!(tree, embedding_dims[index], GaussianPartition())
        retrieve_states_continuous!(tree, embedding_landmark)
        diffusions = optimize_diffusions(tree, embedding_dims[index], gss_lower_bound, gss_upper_bound_diffusion, error_tolerance)
        models = [BrownianMotion(0.0, diffusions[i]) for i in 1:embedding_dims[index]]
        push!(marginal_state_dictionaries_landmark, marginal_state_dict(tree, models))
    end


    # --- Accuracy measures ---
    PiQ_root_probs = get_probabilities(PiQ_model_dictionary[tree])
    real_Q_root_probs = get_probabilities(real_Q_dictionary[tree])

    mds_root_probs = []
    landmark_root_probs = []


    for index in 1:length(embedding_dims)
        push!(mds_root_probs, get_probabilities(marginal_state_dictionaries_MDS[index][tree], embeddings_MDS[index]))
        push!(landmark_root_probs, get_probabilities(marginal_state_dictionaries_landmark[index][tree], embeddings_landmark[index]))
   end


    PiQ_prob_of_start_index = PiQ_root_probs[start_index]
    real_Q_prob_of_start_index = real_Q_root_probs[start_index]
    push!(PiQ_prob_of_start_indices, PiQ_prob_of_start_index)
    push!(real_q_prob_of_start_indices, real_Q_prob_of_start_index)

    
    for index in 1:length(embedding_dims)
        push!(mds_prob_of_start_indices[index], mds_root_probs[index][start_index])
        push!(landmark_prob_of_start_indices[index], landmark_root_probs[index][start_index])
    end

    # ---  KL divergence (forward) ---
    PiQ_kl_div_forward = kl_divergence(real_Q_root_probs, PiQ_root_probs)
    push!(PiQ_kl_divergences_forward, PiQ_kl_div_forward)

    for index in 1:length(embedding_dims)
        push!(mds_kl_divergences_forward[index], kl_divergence(real_Q_root_probs, mds_root_probs[index]))
        push!(landmark_kl_divergences_forward[index], kl_divergence(real_Q_root_probs, landmark_root_probs[index]))
    end

    # --- KL divergence (backward) ---
    PiQ_kl_div_backward = kl_divergence(PiQ_root_probs, real_Q_root_probs)
    push!(PiQ_kl_divergences_backward, PiQ_kl_div_backward)
    for index in 1:length(embedding_dims)
        push!(mds_kl_divergences_backward[index], kl_divergence(mds_root_probs[index], real_Q_root_probs))
        push!(landmark_kl_divergences_backward[index], kl_divergence(landmark_root_probs[index], real_Q_root_probs))
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
   
    PIQ_KL_FORWARD_DIVS = PiQ_kl_divergences_forward,
    MDS_KL_FORWARD_DIVS = format_result(mds_kl_divergences_forward),
    LANDMARK_KL_FORWARD_DIVS = format_result(landmark_kl_divergences_forward),
    
    PIQ_KL_BACKWARD_DIVS = PiQ_kl_divergences_backward,
    MDS_KL_BACKWARD_DIVS = format_result(mds_kl_divergences_backward),
    LANDMARK_KL_BACKWARD_DIVS = format_result(landmark_kl_divergences_backward),

    REAL_Q_ROOT_PROB_OF_START_INDICES = real_q_prob_of_start_indices,
    PIQ_ROOT_PROB_OF_START_INDICES = PiQ_prob_of_start_indices,
    MDS_ROOT_PROB_OF_START_INDICES = format_result(mds_prob_of_start_indices),
    LANDMARK_ROOT_PROB_OF_START_INDICES = format_result(landmark_prob_of_start_indices),
    
    MINIMUM_MIGRATIONS = minimum_migrations,
    TOTAL_MIGRATIONS = total_migrations,
    AVERAGE_MIGRATIONS = average_migrations,
    TOTAL_BRANCHLENGTHS = total_branchlengths,
    DIMS = [embedding_dims for _ = 1:iterations]
)

# --- Write to file ---
cd(@__DIR__)

prefix = "low_dim_results_"

if use_flights 
    global prefix *= "flights_"
end

filename = string(prefix, Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), ".csv")
CSV.write(filename, data)
