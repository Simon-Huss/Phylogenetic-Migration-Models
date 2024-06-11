include("../utils/load.jl")

#=
    File to generate an animation of the migration process on a tree to show the migration process and visualize the impact the parameters have on the migration rate.
=#

dim = 500 # number of cities
number_of_neighbours_knn = 10 # Number of neighbours in the KNN algorithm
use_standard_MDS = true # Standard MDS if true, else stochastic MDS
use_flights = true # Use flights in the distance matrix D and transition matrix Q if true, else use only geodesic distance and local migration
flight_cities = 10
start_index = 28 # Index of the starting city
prop_const_alpha = 0.01 # Transition rate proportionality constant, lower value means less migration and takes longer to simulate

M = get_coords_2D(dim, return_name=false)

knn_tree = KDTree(M)
K = knn(knn_tree, M, number_of_neighbours_knn+1)


# --- Setting up model for simulation ---
# Initial state
X_0 = zeros(dim)
X_0[start_index] = 1

partition = Partition_CTMC(X_0)
model = Gillespie_CTMC(K, (args...; kwargs...) -> KNN_migration(args..., prop_const_alpha, flight_cities, false, use_flights))


println("Generating tree")
tree = sim_tree(1000, 1000.0, 0.1)
initialize_node_data(tree)
println("Tree generated")

println("Initializing tree")
internal_message_init!(tree, partition)

# Sampling the states for the tree nodes
println("Sampling starting")
sampling_time = @elapsed begin
sample_down!(tree, model)
end
println("Sampling done in ", sampling_time, " seconds with a total migration of: ", model.counter.value)

# Saving states for transfer to continuous model
save_actual_states!(tree)
 

max_len = 0
for node in getleaflist(tree)
    current_height = getdistfromroot(node)
    if current_height > max_len
        global max_len = current_height
    end
end

tree_height = max_len

frames = 250

frame_time = tree_height/frames

# Pre-calculate all the heights
height_dictionary = Dict()
[height_dictionary[node] = getdistfromroot(node) for node in getnodelist(tree)]

function get_counter_colours(candidates, colors, i)
    t = i * frame_time

    new_nodes = []
    new_candidates = []
    while candidates != []
        candidate = pop!(candidates)
        height = height_dictionary[candidate]
        if height < t
            push!(new_nodes, candidate)
            for child in candidate.children
                push!(candidates, child)
            end
        else
            push!(new_candidates, candidate)
        end
    end

    
    for node in new_nodes
        colors[node.node_data["index"]] = RGB(1, 0, 0)        
    end
    return colors, new_candidates
end

candidates = [tree]
colors = repeat([RGB(0, 0, 1)], dim)

anim = @animate for i in 1:frames
    global colors, candidates = get_counter_colours(candidates, colors, i)
    Plots.scatter(M[1, :], M[2, :], color=colors, label="", title=string("Frame: ", i), markerstrokewidth=0, markersize=1.5)
end

gif(anim, "test.gif", fps = 15)

