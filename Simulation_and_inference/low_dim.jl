include("../utils/load.jl")

#=
    This file outlines the approach to simulate and infer the continuous-time Markov chain on a phylogenetic tree for the different models and embeddings.
    Can be used to test out different parameter values to see the effect on the simulation and inference process before running the high dimensional simulations or the result files.
=#

total_time = @elapsed begin
number_of_cities = 250 # Number of cities i.e dimensionality of the data
number_of_neighbours_knn = 5 # Number of neighbours in the KNN algorithm

use_flights = true # Use flights in the distance matrix D and transition matrix Q if true, else use only geodesic distance and local migration
flight_cities = 10 # Number of cities with flights

embedding_dim = 5 # Embedding dimension i.e number of dimensions in our embedded vector space
initial_mds_datapoints = 20 # Number of dimensions in the initial MDS embedding

start_index = 200 # Index of the starting city
prop_const_alpha = 0.000_01 # Transition rate proportionality constant, lower value means less migration and takes longer to simulate

# Diffusion GSS search parameters
gss_lower_bound = 0
gss_upper_bound_rscaling = 2
gss_upper_bound_diffusion = 200
error_tolerance = 1e-6


coordinates, cities, countries = get_coords(number_of_cities, return_names=true, return_countries=true)

# --- Tree ---
tree = sim_tree(2500, 1000.0, 0.05)
initialize_node_data(tree) # Initializes the node data dictionaries for the tree
#MolecularEvolution.simple_tree_draw(tree, line_color="white")  # Used to visualize the phylogenetic tree


# --- MDS & Landmark MDS --- 
embedding_MDS = MDS_embedding(coordinates, embedding_dim, use_flights, flight_cities)
embedding_landmark = Landmark_MDS(coordinates, embedding_dim, use_flights, flight_cities, initial_mds_datapoints) 

# --- KNN ---
knn_tree = KDTree(coordinates)
K = knn(knn_tree, coordinates, number_of_neighbours_knn+1)

# --- Setting up model for simulation ---
println("Starting city: ", cities[start_index])
partition = Partition_CTMC(start_index)
internal_message_init!(tree, partition)
model = Gillespie_CTMC(K, (args...; kwargs...) -> KNN_migration(args..., prop_const_alpha, flight_cities, false, use_flights))

# --- Sampling the states for the tree nodes ---
sampling_time = @elapsed begin
    sample_down!(tree, model)
end
println("Sampling done in ", sampling_time, " seconds.")
save_actual_states!(tree, cities)

# --- PiQ with ones inference ---
uniform_init!(tree, number_of_cities, CustomDiscretePartition(number_of_cities, 1))
retrieve_states_discrete!(tree)
ones_model = PiQ(ones(number_of_cities))
PiQ_model, opt_r = optimize_rscaling(ones_model, tree, gss_lower_bound, gss_upper_bound_rscaling)
PiQ_model_dictionary = marginal_state_dict(tree, PiQ_model)


# --- Real Q inference ---
Q = zeros(number_of_cities, number_of_cities);
for i=1:number_of_cities
    Q[i,:] = KNN_migration(K, i, prop_const_alpha, flight_cities, true, use_flights) # Build the transition matrix Q using the KNN migration function with the "get Q" parameter set to true
end
uniform_init!(tree, number_of_cities, CustomDiscretePartition(number_of_cities, 1))
retrieve_states_discrete!(tree)
real_Q_model = GeneralCTMC(Q)
real_Q_dictionary = marginal_state_dict(tree, real_Q_model)

# --- Continuous embedding MDS inference ---
uniform_init!(tree, embedding_dim, GaussianPartition())
retrieve_states_continuous!(tree, embedding_MDS)
mds_diffusions, mds_lls = optimize_diffusions(tree, embedding_dim, gss_lower_bound, gss_upper_bound_diffusion, error_tolerance, return_lls=true)
models = [BrownianMotion(0.0, mds_diffusions[i]) for i in 1:embedding_dim]
marginal_state_dictionary_MDS = marginal_state_dict(tree, models)

# --- Continuous embedding landmark MDS inference ---
uniform_init!(tree, embedding_dim, GaussianPartition())
retrieve_states_continuous!(tree, embedding_landmark)
landmark_diffusions, landmark_lls = optimize_diffusions(tree, embedding_dim, gss_lower_bound, gss_upper_bound_diffusion, error_tolerance, return_lls=true)
models = [BrownianMotion(0.0, landmark_diffusions[i]) for i in 1:embedding_dim]
marginal_state_dictionary_landmark = marginal_state_dict(tree, models)

# --- Accuracy tests ---
println("Starting accuracy tests:")
PiQ_root_probs = get_probabilities(PiQ_model_dictionary[tree])
real_Q_root_probs = get_probabilities(real_Q_dictionary[tree])
mds_root_probs = get_probabilities(marginal_state_dictionary_MDS[tree], embedding_MDS)
landmark_root_probs = get_probabilities(marginal_state_dictionary_landmark[tree], embedding_landmark)

PiQ_kl_div_forward = kl_divergence(real_Q_root_probs, PiQ_root_probs)
mds_kl_div_forward = kl_divergence(real_Q_root_probs, mds_root_probs)
landmark_kl_div_forward = kl_divergence(real_Q_root_probs, landmark_root_probs)

PiQ_kl_div_backward = kl_divergence(PiQ_root_probs, real_Q_root_probs)
mds_kl_div_backward = kl_divergence(mds_root_probs, real_Q_root_probs)
landmark_kl_div_backward = kl_divergence(landmark_root_probs, real_Q_root_probs)


# --- Tree probability ---
real_Q_TP = tree_probability(tree, real_Q_dictionary)
PiQ_TP = tree_probability(tree, PiQ_model_dictionary)
mds_TP = tree_probability(tree, marginal_state_dictionary_MDS, embedding_MDS)
landmark_TP = tree_probability(tree, marginal_state_dictionary_landmark, embedding_landmark)

# --- Root probability for each model ---
real_Q_root_prob = real_Q_root_probs[start_index]
PiQ_root_prob = PiQ_root_probs[start_index]
mds_root_prob = mds_root_probs[start_index]
landmark_root_prob = landmark_root_probs[start_index]

#=
# --- KNN Root probability for each model ---
# The node_probability_knn is similar to root probability except it also includes the neighboring states in the probability
real_Q_knn_root_prob = node_probability_knn(tree, real_Q_dictionary, K)
PiQ_knn_root_prob = node_probability_knn(tree, PiQ_model_dictionary, K)
mds_knn_root_prob = node_probability_knn(tree, marginal_state_dictionary_MDS, embedding_MDS, K)
landmark_knn_root_prob = node_probability_knn(tree, marginal_state_dictionary_landmark, embedding_landmark, K)
=#

println()
println("Forward KL divergence PiQ: ", PiQ_kl_div_forward)
println("Forward KL divergence MDS: ", mds_kl_div_forward)
println("Forward KL divergence Landmark: ", landmark_kl_div_forward)
println()
println("Backward KL divergence PiQ: ", PiQ_kl_div_backward)
println("Backward KL divergence MDS: ", mds_kl_div_backward)
println("Backward KL divergence Landmark: ", landmark_kl_div_backward)
println()
println("Tree probability PiQ: ", PiQ_TP)
println("Tree probability MDS: ", mds_TP)
println("Tree probability Landmark: ", landmark_TP)
println("Tree probability Real Q: ", real_Q_TP)
println()
println("Root probability PiQ: ", PiQ_root_prob)
println("Root probability MDS: ", mds_root_prob)
println("Root probability Landmark: ", landmark_root_prob)
println("Root probability Real Q: ", real_Q_root_prob)
println()
#=
println("KNN Root probability PiQ: ", PiQ_knn_root_prob)
println("KNN Root probability MDS: ", mds_knn_root_prob)
println("KNN Root probability Landmark: ", landmark_knn_root_prob)
println("KNN Root probability Real Q: ", real_Q_knn_root_prob)
println()
=#

sort_probabilities = true
bars = 20
root_plot_MDS = get_node_plot(tree, marginal_state_dictionary_MDS, cities, embedding_MDS, bars, "MDS", sort_probabilities)
root_plot_landmark = get_node_plot(tree, marginal_state_dictionary_landmark, cities, embedding_landmark, bars, "Landmark MDS", sort_probabilities)
root_plot_PiQ = get_node_plot(tree, PiQ_model_dictionary, cities, bars, "PiQ", sort_probabilities)
root_plot_realQ = get_node_plot(tree, real_Q_dictionary, cities, bars, "real Q", sort_probabilities)
Plots.plot(root_plot_MDS, root_plot_landmark, root_plot_PiQ, root_plot_realQ, layout = (2,2))

PiQ_country_probs = country_probabilities(PiQ_root_probs, countries)
real_Q_country_probs = country_probabilities(real_Q_root_probs, countries)
mds_country_probs = country_probabilities(mds_root_probs, countries)
landmark_country_probs = country_probabilities(landmark_root_probs, countries)

actual_country = countries[start_index]
country_root_plot_MDS = get_country_plot(mds_country_probs, "MDS", actual_country = actual_country)
country_root_plot_Landmark = get_country_plot(landmark_country_probs, "Landmark MDS", actual_country = actual_country)
country_root_plot_PiQ = get_country_plot(PiQ_country_probs, "PiQ", actual_country = actual_country)
country_root_plot_RealQ = get_country_plot(real_Q_country_probs, "Real Q", actual_country = actual_country)
Plots.plot(country_root_plot_MDS, country_root_plot_Landmark, country_root_plot_PiQ, country_root_plot_RealQ, layout = (2,2))

end
println("Total time: ", total_time)

# Plotting the landmark embedding diffusion likelihoods to visualize the optimization process
#=plts = []
for i = 1:embedding_dim
    x_values = range(landmark_diffusions[i]/2, stop=landmark_diffusions[i]*2, length=1000)
    push!(plts, Plots.plot(x_values, landmark_lls[i].(x_values), legend=false))
end
Plots.plot(plts..., layout=(1,embedding_dim), size=(3000, 1500), title="Landmark diffusions")
=#