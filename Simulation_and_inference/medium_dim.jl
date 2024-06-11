include("../utils/load.jl")

#=
    This file increases the dimensionality and complexity of the simulation and inference process compared to the low dimensional case in low_dim_jl.
=#

total_time = @elapsed begin
number_of_cities = 10_000 # Number of cities i.e dimensionality of the data
number_of_neighbours_knn = 200 # Number of neighbours in the KNN algorithm

use_flights = false # Use flights in the distance matrix D and transition matrix Q if true, else use only geodesic distance and local migration
flight_cities = 400 # Number of cities with flights

embedding_dim = 25 # Embedding dimension i.e number of dimensions in our embedded vector space
initial_mds_datapoints = 800 # Number of dimensions in the initial MDS embedding

start_index = 17 # Index of the starting city
prop_const_alpha = 2.5*1e-7 # Transition rate proportionality constant, lower value means less migration and takes longer to simulate

# Diffusion GSS search parameters
gss_lower_bound = 0
gss_upper_bound_rscaling = 2
gss_upper_bound_diffusion = 200
error_tolerance = 1e-6

coordinates, cities, countries = get_coords(number_of_cities, return_names=true, return_countries=true)

# --- Tree ---
tree = sim_tree(100_000, 1000.0, 0.05)
initialize_node_data(tree) # Initializes the node data dictionaries for the tree
# MolecularEvolution.simple_tree_draw(tree, line_color="white")  # Used to visualize the phylogenetic tree


# --- MDS --- 
mds_time = @elapsed begin
    embedding_MDS = MDS_embedding(coordinates, embedding_dim, use_flights, flight_cities)
end
println("MDS done in ", mds_time, " seconds.")

# --- Landmark MDS ---
landmark_time = @elapsed begin
    embedding_landmark = Landmark_MDS(coordinates, embedding_dim, use_flights, flight_cities, initial_mds_datapoints) 
end
println("Landmark MDS done in ", landmark_time, " seconds.")

# --- KNN ---
knn_time = @elapsed begin
    knn_tree = KDTree(coordinates)
    K = knn(knn_tree, coordinates, number_of_neighbours_knn+1)
end
println("KNN done in ", knn_time, " seconds.")

# --- Setting up model for simulation ---
println("Starting city: ", cities[start_index])
initialization_time = @elapsed begin
    partition = Partition_CTMC(start_index)
    internal_message_init!(tree, partition)
end
println("Initialization done in ", initialization_time, " seconds.")

model = Gillespie_CTMC(K, (args...; kwargs...) -> KNN_migration(args..., prop_const_alpha, flight_cities, false, use_flights))
# Sampling the states for the tree nodes
sampling_time = @elapsed begin
    @time sample_down!(tree, model)
end
println("Sampling done in ", sampling_time, " seconds.")


save_time = @elapsed begin
    save_actual_states!(tree, cities)    
end
println("Save states done in ", save_time, " seconds.")

migrations = [x.node_data["migrations_from_root"] for x in getleaflist(tree)]

bl = branchlengths(tree)
tm = model.counter.value
println("total migrations ", tm)
println("total_branchlengths ", bl)
println("average_migrations ", tm/bl)
println("minimum_migrations ", minimum(migrations))

println("PIQ")
piq_time = @elapsed begin
# --- PiQ with ones inference ---
uniform_init!(tree, number_of_cities, CustomDiscretePartitionFloat32(number_of_cities, 1))
retrieve_states_discrete!(tree)
ones_model = PiQ(ones(number_of_cities))
PiQ_model = PiQ(ones(number_of_cities))
r_scale_time = @elapsed begin
   PiQ_model, opt_r = optimize_rscaling(ones_model, tree, gss_lower_bound, gss_upper_bound_rscaling)
end
println("R scaling done in ", r_scale_time, " seconds.")
piq_inferrence_time = @elapsed begin
@time PiQ_model_dictionary = marginal_state_dict(tree, PiQ_model)
end
println("PiQ inferrence done in ", piq_inferrence_time, " seconds.")
end
println("PiQ done in ", piq_time, " seconds.")


# --- continuous embedding MDS inference ---
mds_inferrence_time = @elapsed begin
uniform_init!(tree, embedding_dim, GaussianPartition())
retrieve_states_continuous!(tree, embedding_MDS)
mds_diffusions, mds_lls = optimize_diffusions(tree, embedding_dim, gss_lower_bound, gss_upper_bound_diffusion, error_tolerance, return_lls=true)
models = [BrownianMotion(0.0, mds_diffusions[i]) for i in 1:embedding_dim]
marginal_state_dictionary_MDS = marginal_state_dict(tree, models)
end
println("MDS inferrence done in ", mds_inferrence_time, " seconds.")


# --- continuous embedding landmark MDS inference ---
landmark_inferrence_time = @elapsed begin
uniform_init!(tree, embedding_dim, GaussianPartition())
retrieve_states_continuous!(tree, embedding_landmark)
landmark_diffusions, landmark_lls = optimize_diffusions(tree, embedding_dim, gss_lower_bound, gss_upper_bound_diffusion, error_tolerance, return_lls=true)
models = [BrownianMotion(0.0, landmark_diffusions[i]) for i in 1:embedding_dim]
@time marginal_state_dictionary_landmark = marginal_state_dict(tree, models)
end
println("Landmark inferrence done in ", landmark_inferrence_time, " seconds.")


println("Starting accuracy tests:")
get_prob_times = @elapsed begin
PiQ_root_probs = get_probabilities(PiQ_model_dictionary[tree])
mds_root_probs = get_probabilities(marginal_state_dictionary_MDS[tree], embedding_MDS)
landmark_root_probs = get_probabilities(marginal_state_dictionary_landmark[tree], embedding_landmark)
end

println("Get probabilities done in ", get_prob_times, " seconds.")
println()
println("Root probability PiQ: ", PiQ_root_probs[start_index])
println("Root probability MDS: ", mds_root_probs[start_index])
println("Root probability Landmark: ", landmark_root_probs[start_index])
println()

#= 
# WARNING, tree_probability takes a long time to run for large trees! Consider specifying a subset_size if you want to test this on a random subset of nodes
tree_prob_time = @elapsed begin
PiQ_TP = tree_probability(tree, PiQ_model_dictionary)
mds_TP = fast_tree_probability(tree, marginal_state_dictionary_MDS, embedding_MDS)
landmark_TP = fast_tree_probability(tree, marginal_state_dictionary_landmark, embedding_landmark)
end
println("Tree prob time: ", tree_prob_time)
println()
println("Tree probability PiQ: ", PiQ_TP)
#println("Tree probability MDS: ", mds_TP)
println("Tree probability Landmark: ", landmark_TP)
println()
=#

end
println("Total time: ", total_time)
