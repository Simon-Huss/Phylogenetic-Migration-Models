include("../utils/load.jl")

#= 
    Note: 
    The likelihood function for the r scaling parameter in the PiQ model is not convex and there are for some setups a very narrow spike for certain values which can easily be missed as the likelihood function flattens out
    quickly so a simple optimization algorithm might miss the true optimum and extra care has to be taken (in our case solved by the golden section search algorithm preferring the left interval if both are mutually beneficial).

    This file also highlights the importance of the r-scaling being performed by comparing the tree probability (average node probability across all internal nodes) of the PiQ model with and without the r-scaling optimization.
=#

number_of_cities = 100 
number_of_neighbours_knn = 5

use_flights = false
flight_cities = 10

start_index = 28
prop_const_alpha = 0.0001;

X_0 = zeros(number_of_cities, 1);
X_0[start_index] = 1;

gss_low = 1e-6
gss_high = 1
error_tolerance = 1e-10

# --- Setting up the tree ---
tree = sim_tree(2500, 1000.0, 0.5);
initialize_node_data(tree)


# --- Creating KNN ---
coords, cities = get_coords(number_of_cities, return_names=true)
knn_tree = KDTree(coords)
K = knn(knn_tree, coords, number_of_neighbours_knn+1)

println("Starting city: ", cities[start_index])

Q = zeros(number_of_cities, number_of_cities);
for i=1:number_of_cities
    Q[i,:] = KNN_migration(K, i, prop_const_alpha, flight_cities, true, use_flights) 
end

# --- Assigning the real Q to the model ---
real_Q_model = GeneralCTMC(Q)

PiQ_model = PiQ(ones(number_of_cities))

partition = CustomDiscretePartition(number_of_cities, 1);
partition.state[:,1] = ones(number_of_cities)

println("Initializing tree")
internal_message_init!(tree, partition)


# Sampling the states for the tree nodes
println("Sampling starting")
tree.parent_message[1].state = X_0 # Setting prior for root for the sampling

sampling_time = @elapsed begin
    sample_down!(tree, real_Q_model)
end

save_actual_states!(tree, cities)

optimized_PiQ_model, r = optimize_rscaling(PiQ_model, tree, gss_low, gss_high, error_tolerance)

println("Sampling done in ", sampling_time, " seconds.\n")
tree.parent_message[1].state = ones(number_of_cities, 1)
optimized_PiQ_model_dictionary = marginal_state_dict(tree, optimized_PiQ_model);

tree.parent_message[1].state = ones(number_of_cities, 1)
real_Q_dictionary = marginal_state_dict(tree, real_Q_model);

tree.parent_message[1].state = ones(number_of_cities, 1)
PiQ_model_dictionary = marginal_state_dict(tree, PiQ_model);


println("PiQ tree probability: ", tree_probability(tree, PiQ_model_dictionary))
println("Optimized PiQ tree probability: ", tree_probability(tree, optimized_PiQ_model_dictionary))
println("Real Q tree probability: ", tree_probability(tree, real_Q_dictionary))

pi = eq_freq(PiQ_model)

ll(x) = log_likelihood!(tree, PiQ(x, pi))
x_values = range(gss_low, gss_high, length=1000)
plot(x_values, ll.(x_values), label="L(r)", xlabel="r", ylabel="Log-likelihood L(r)", title="R-scaling optimization for the PiQ model")
scatter!([r], [ll.(r)], label="r*")
