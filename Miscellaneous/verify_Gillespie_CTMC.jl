include("../utils/load.jl")

#= 
    This script is used to verify the correctness of the continuous-time Markov chain (CTMC) sampling implementation for the Gillespie_CTMC model.
    The script generates a random Q-matrix and a random tree, and then samples down the tree to estimate the state distribution at each node.
    The analytical solution is compared to the Monte-Carlo estimate.
=#

# Parameters, note iterations needs to be large to reduce the statistical error (but it will take a long time to run)
iterations = 1000

dimensions = 5

Q = rand(dimensions, dimensions).*50

for i in 1:size(Q,1)
    global Q[i,i] = 0
    global Q[i,i] = -sum(Q[i,:])
end

model = Gillespie_CTMC(Q, (Q, X; kwargs...) -> Q_to_J(Q[X, :], X))

X_0 = zeros(dimensions)
# Works for both determinstic and stochastic initial distributions
#X_0[1] = 1
X_0 = [0.3, 0.1, 0.25, 0.2, 0.15]


# Generate a FelNode directly with sim_tree
tree = sim_tree(100, 100.0, 0.5)
initialize_node_data(tree)

# Specify partitions and branchmodel
partitions = Partition_CTMC(X_0)  # Partition with initial state X_0
part_models = [model]

internal_message_init!(tree, partitions)

# Initialize dictionary with zeros
node_dict = Dict{FelNode, Array{Float64, 1}}()
for node in getnodelist(tree)
    node_dict[node] = zeros(size(Q)[1])
end

t = @elapsed begin
    # Sample down to generate node data for i iterations and save the outcomes in the dictionary as a Monte-Carlo estimate 
    for i=1:iterations
        tree.message[1].state = X_0
        sample_down!(tree, part_models)
        for node in getnodelist(tree)
            node_dict[node] += node.message[1].state
        end 
    end
end #end time

# Print the results and compare to the analytical solution exp(Q*T)*X_0, where T = branchlength and X_0 = initial distribution
difference = 0
for node in getnodelist(tree)
    T = getdistfromroot(node)
    analytical = LinearAlgebra.exp!(Q*T)'X_0
    global difference += norm(node_dict[node]/iterations - analytical)
end

println("Total difference: ", difference, " iterations: ", iterations, " dimensions: ", dimensions, " time: ", t)

