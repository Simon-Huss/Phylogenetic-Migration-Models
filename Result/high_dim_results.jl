include("../utils/load.jl")

# OBS! The geonames dataset is cut at 10_000 cities, download the entire dataset to run on more cities.

iterations = 1
full_total_time = @elapsed begin

number_of_cities = 145_000 # Number of cities i.e dimensionality of the data
number_of_neighbours_knn = 3000 # Number of neighbours in the KNN algorithm

use_flights = true # Use flights in the distance matrix D and transition matrix Q if true, else use only geodesic distance and local migration
flight_cities = 6000 # Number of cities with flights

embedding_dims = [1, 2, 3, 5, 10]
initial_mds_datapoints = 12_000 # Number of dimensions in the initial MDS embedding for the landmark algorithm
prop_const_alpha = 5/3*1e-8 # Transition rate proportionality constant, lower value means less migration and takes longer to simulate

# ---  GSS search parameters ---
gss_lower_bound = 0
gss_upper_bound_rscaling = 2
gss_upper_bound_diffusion = 200
error_tolerance = 1e-6

coordinates, cities, countries = get_coords(number_of_cities, return_names=true, return_countries=true)

landmark_diffusion_times = [[] for _ in embedding_dims]
landmark_inference_times = [[] for _ in embedding_dims]
landmark_total_times = [[] for _ in embedding_dims]

minimum_migrations = []
average_migrations = []
total_branchlengths = []
total_migrations = []

# --- KNN ---
knn_tree = KDTree(coordinates)
K = knn(knn_tree, coordinates, number_of_neighbours_knn+1)


# --- Landmark MDS --- 
embeddings_landmark = []
for embedding_dim in embedding_dims
    push!(embeddings_landmark, Landmark_MDS(coordinates, embedding_dim, use_flights, flight_cities, initial_mds_datapoints)) #MDS_embedding(false, coordinates, embedding_dim, use_flights, flight_cities, initial_mds_datapoints, iterations, learning_rate)
end


for i = 1:iterations

    println((i-1)/iterations*100, "%")
    
    start_index = rand(1:number_of_cities)

    # --- Tree ---
    println("Creating tree ", now())
    local tree = sim_tree(1_000_000, 1000.0, 0.05)
    initialize_node_data(tree)
    
    #--- Setting up model for simulation ---
    println("Sampling down ", now())
    local partition = Partition_CTMC(start_index)
    internal_message_init!(tree, partition)
    local model = Gillespie_CTMC(K, (args...; kwargs...) -> KNN_migration(args..., prop_const_alpha, flight_cities, false, use_flights))

    # ---  Sampling the states for the tree nodes ---
    sample_down!(tree, model)
    save_actual_states!(tree, cities)

    # --- Tree statistics ---
    migrations = [x.node_data["migrations_from_root"] for x in getleaflist(tree)]
    bl = branchlengths(tree)
    tm = model.counter.value
    push!(total_migrations, tm)
    push!(total_branchlengths, bl)
    push!(average_migrations, tm/bl)
    push!(minimum_migrations, minimum(migrations))

    println("Starting landmark model ", now())

    # --- Continuous embedding landmark MDS inference ---
    for (index, embedding_landmark) in enumerate(embeddings_landmark)
        total_time = @elapsed begin
            uniform_init!(tree, embedding_dims[index], GaussianPartition())
            retrieve_states_continuous!(tree, embedding_landmark)
            diff_time = @elapsed begin
                diffusions = optimize_diffusions(tree, embedding_dims[index], gss_lower_bound, gss_upper_bound_diffusion, error_tolerance)
            end 
            push!(landmark_diffusion_times[index], diff_time)
            models = [BrownianMotion(0.0, diffusions[i]) for i in 1:embedding_dims[index]]
            inference_time = @elapsed begin
                d = marginal_state_dict(tree, models)
            end
            push!(landmark_inference_times[index], inference_time)
        end
        push!(landmark_total_times[index], total_time)
        println("Embedding dimension ", embedding_dims[index], " done ", now())
    end

    # --- Accuracy measures ---
    #= Commented out for speed purposes, but can be included if needed, just make sure to uncomment the code below as well in the DataFrame creation! 
    println("Starting accuracy measurements ", now())

    landmark_root_probs = []

    for index in 1:length(embedding_dims)
        push!(landmark_root_probs, fast_get_probabilities(marginal_state_dictionaries_landmark[index][tree], embeddings_landmark[index]))
    end
    
    for index in 1:length(embedding_dims)
        push!(landmark_prob_of_start_indices[index], landmark_root_probs[index][start_index])
    end
    =#
    

    
   
end # for loop
end # total time

println("100.0%")

println("Total time: ", full_total_time, " and time per iteration: ", full_total_time/iterations)

data = DataFrame(
    
    #LANDMARK_ROOT_PROB_OF_START_INDICES = format_result(landmark_prob_of_start_indices),
    LANDMARK_DIFFUSION_TIMES = format_result(landmark_diffusion_times),
    LANDMARK_INFERENCE_TIMES = format_result(landmark_inference_times),
    LANDMARK_TOTAL_TIMES = format_result(landmark_total_times),
    MINIMUM_MIGRATIONS = minimum_migrations,
    TOTAL_MIGRATIONS = total_migrations,
    AVERAGE_MIGRATIONS = average_migrations,
    TOTAL_BRANCHLENGTHS = total_branchlengths,
    DIMS = [embedding_dims for _ = 1:iterations]
)

# ---  Write to file ---
cd(@__DIR__)

prefix = "high_dim_results_"

if use_flights 
    global prefix *= "flights_"
end

filename = string(prefix, Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), ".csv")
CSV.write(filename, data)
