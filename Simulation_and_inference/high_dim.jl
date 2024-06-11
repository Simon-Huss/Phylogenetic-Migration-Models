include("../utils/load.jl")

#=
    Note! The geonames dataset is cut at 10_000 cities, download the entire dataset to run on more cities.
    This file illustrates that the approach using the landmark embedding scales to high dimensions and large datasets.
=#
total_time = @elapsed begin
number_of_cities = 145_000 # Number of cities i.e dimensionality of the data
number_of_neighbours_knn = 3000 # Number of neighbours in the KNN algorithm

use_flights = true # Use flights in the distance matrix D and transition matrix Q if true, else use only geodesic distance and local migration
flight_cities = 6000 # Number of cities with flights

embedding_dim = 5 # Embedding dimension i.e number of dimensions in our embedded vector space
initial_mds_datapoints = 12_000 # Number of dimensions in the initial MDS embedding

start_index = 17 # Index of the starting city
prop_const_alpha = 5/3*1e-8 # Transition rate proportionality constant, lower value means less migration but faster to simulate


# Diffusion GSS search parameters
gss_lower_bound = 0
gss_upper_bound_rscaling = 2
gss_upper_bound_diffusion = 200
error_tolerance = 1e-6

coordinates, cities, countries = get_coords(number_of_cities, return_names=true, return_countries=true)

# --- Tree ---
println("Tree generation starting. ", now())
@time tree = sim_tree(1_000_000, 1000.0, 0.05)
initialize_node_data(tree) # Initializes the node data dictionaries for the tree
# MolecularEvolution.simple_tree_draw(tree, line_color="white") # Used to visualize the phylogenetic tree, too large for this amount of nodes however a subtree can be drawn instead if needed
println("Tree generated. ", now())

# --- Landmark MDS ---
landmark_time = @elapsed begin
    embedding_landmark = Landmark_MDS(coordinates, embedding_dim, use_flights, flight_cities, initial_mds_datapoints)
end
println("Landmark MDS done in ", landmark_time, " seconds.", now())

# --- KNN ---
knn_time = @elapsed begin
knn_tree = KDTree(coordinates)
K = knn(knn_tree, coordinates, number_of_neighbours_knn+1)
end
println("KNN done in ", knn_time, " seconds. ", now())

# --- Setting up model for simulation ---
println("Starting city: ", cities[start_index])
initialization_time = @elapsed begin
partition = Partition_CTMC(start_index)
internal_message_init!(tree, partition)
end
println("Initialization done in ", initialization_time, " seconds. ", now())
model = Gillespie_CTMC(K, (args...; kwargs...) -> KNN_migration(args..., prop_const_alpha, flight_cities, false, use_flights))

# --- Sampling the states for the tree nodes ---
sampling_time = @elapsed begin
    @time sample_down!(tree, model)
end
println("Sampling done in ", sampling_time, " seconds. ", now())

save_time = @elapsed begin
    save_actual_states!(tree, cities)    
end
println("Save states done in ", save_time, " seconds. ", now())


# --- Continuous embedding landmark MDS inference ---
landmark_inferrence_time = @elapsed begin
uniform_init!(tree, embedding_dim, GaussianPartition())
retrieve_states_continuous!(tree, embedding_landmark)
landmark_diffusions, landmark_lls = optimize_diffusions(tree, embedding_dim, gss_lower_bound, gss_upper_bound_diffusion, error_tolerance, return_lls=true)
models = [BrownianMotion(0.0, landmark_diffusions[i]) for i in 1:embedding_dim]
marginal_state_dictionary_landmark = marginal_state_dict(tree, models)
end
println("Landmark inferrence done in ", landmark_inferrence_time, " seconds.")

end # Total time
println("Total time: ", total_time)
