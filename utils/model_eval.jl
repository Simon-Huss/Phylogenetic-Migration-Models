# Gets the continuous embedded probabilities of a states marginal_state_dictionary entry
function get_probabilities(marginal_state, embedding)
    
    means = [x.mean for x in marginal_state]
    vars = sqrt.([x.var for x in marginal_state])
    
    # Below line is a double "for loop", one loop over all the dimensions of the embedding (rows of the embedding) and a second "for loop" over all cities (columns of the embedding).
    # It is then summed over the columns (i.e dims = 1) to get all the embedding dimensions for each city resulting in a vector of logarithmic likelihoods for each city
    logprobabilities = sum(logpdf.(Normal.(means, vars), embedding), dims=1)
    # Renormalizing into probabilities
    translation = maximum(logprobabilities)
    logprobabilities .-= translation
    probs = exp.(logprobabilities)
    probs = probs/sum(probs)
    return probability_smoothing(vec(probs))
end

# Uses the get_probabilities function but only returns the probability of the actual state
function get_probability(marginal_state, embedding, actual_state)
    return get_probabilities(marginal_state, embedding)[actual_state] 
end

# Calculates the "tree probability" i.e the average probability of the actual state over all internal nodes (or a random subset of size subset_size) for continuous embeddings
function tree_probability(tree, marginal_state_dictionary, embedding; subset_size = 0)
    prob = 0;
    number_of_nodes = 0
    internal_nodes = getnonleaflist(tree)
    if subset_size != 0
        internal_nodes = sample(internal_nodes, subset_size, replace=false)
    end
    for node in internal_nodes
        actual_state = node.node_data["index"]
        marginal_state = marginal_state_dictionary[node]
        prob += get_probability(marginal_state, embedding, actual_state)
        number_of_nodes += 1
    end
    return prob/number_of_nodes;
end


# Same as tree_probability, but for discrete state-spaces i.e. not using a continuous embedding
function tree_probability(tree, marginal_state_dictionary)
    prob = 0;
    number_of_nodes = 0
    for node in getnonleaflist(tree)
        actual_state = node.node_data["index"]
        prob += marginal_state_dictionary[node][1].state[actual_state]
        number_of_nodes += 1
    end
    return prob/number_of_nodes;
end

# Analog to the continuous get_probabilities but for discrete state-spaces
function get_probabilities(marginal_state)
    return probability_smoothing(vec(marginal_state[1].state))
end

# Retrieves the probability of the root node for continuous state-spaces
function root_probability(tree, marginal_state_dictionary, embedding)
    return get_probability(marginal_state_dictionary[tree], embedding, tree.node_data["index"])
end

# Retrieves the probability of the root node for discrete state-spaces
function root_probability(tree, marginal_state_dictionary)
    return marginal_state_dictionary[tree][1].state[tree.node_data["index"]] 
end

# Save the actual simulated outcomes and the corresponding city names to the tree nodes
function save_actual_states!(tree, cities = nothing)
    for node in getnodelist(tree)
        if (typeof(node.message[1].state) == Int)
            state = node.message[1].state
        else
            state = findfirst(==(1.0), node.message[1].state)[1]
        end
        node.node_data["index"] = state
        if !isnothing(cities)
            node.node_data["city"] = cities[state]
        end
    end
end

# Retreive the leaf node data to use as data for the inference for discrete state-spaces
function retrieve_states_discrete!(tree)
    for leaf in getleaflist(tree)
        leaf.message[1].state[:,1] = zeros(leaf.message[1].states)
        leaf.message[1].state[leaf.node_data["index"], 1] = 1;
    end
end

# Retreive the leaf node data to use as data for the inference for continuous state-spaces
function retrieve_states_continuous!(tree, embedding)
    dim = length(embedding[:,1])
    for node in getleaflist(tree)
        for i=1:dim
            node.message[i].mean = embedding[i, node.node_data["index"]]
            node.message[i].var = 0
        end
    end
end

# Calculates the total branchlength of the tree
function branchlengths(tree::FelNode)
    stack = [tree]
    length = 0
    while !isempty(stack)
       node = pop!(stack)
       length += node.branchlength
       for child in node.children
            push!(stack, child)
       end
    end
    return length
end

# Get the state plots for a node given the marginal state dictionary, the cities and the embedding for the continuous case. The bars parameter is the number of cities to plot, the title is the title of the plot and sort_probs is a boolean to sort the probabilities in descending order or not.
function get_node_plot(node, marginal_state_dictionary, cities, embedding, bars, title, sort_probs)
    probs = get_probabilities(marginal_state_dictionary[node], embedding)
    order = 1:length(cities)
    actual_state = node.node_data["index"]
    if sort_probs
        order = sortperm(probs, rev=true)
        probs = sort(probs, rev=true)
        println(title, " - Index: ", findfirst(==(actual_state), order), " City: ", node.node_data["city"])
    end
    return Plots.bar(string.(cities[order[1:bars]]), (probs[1:bars]), legend=false, xlabel="Locations", xrotation=15, ylabel="Probability", title=title, color = ifelse.(order[1:bars] .== actual_state, :red, :blue))
end

# Get the state plots for a node given the marginal state dictionary and the cities for the discrete case. The bars parameter is the number of cities to plot, the title is the title of the plot and sort_probs is a boolean to sort the probabilities in descending order or not.
function get_node_plot(node, marginal_state_dictionary, cities, bars, title, sort_probs)
    probs = get_probabilities(marginal_state_dictionary[node])
    order = 1:length(cities)
    actual_state = node.node_data["index"]
    if sort_probs
        order = sortperm(probs, rev=true)
        probs = sort(probs, rev=true)
        println(title, " - Index: ", findfirst(==(actual_state), order), " City: ", node.node_data["city"])
    end
    return Plots.bar(string.(cities[order[1:bars]]), (probs[1:bars]), legend=false, xlabel="Locations", xrotation=15, ylabel="Probability", title=title, color = ifelse.(order[1:bars] .== actual_state, :red, :blue))
end

# Get the probability of an entire country
function country_probabilities(probabilities, countries)
    n = length(probabilities)
    country_dict = Dict()

    for i = 1:n
        country = countries[i]
        if haskey(country_dict, country)
            country_dict[country] = (country_dict[country][1] + probabilities[i], country_dict[country][2])
        else
            country_dict[country] = (probabilities[i], country)
        end
    end

    return sort(collect(values(country_dict)), by=first, rev=true)
end

# Get the country plot for a list of probabilities and country names. The bars parameter is the number of countries to plot, the title is the title of the plot and actual_country is the country to highlight in red.
function get_country_plot(tuple_list, title = "Title"; actual_country = nothing, bars=10)
    probs, names = zip(tuple_list...)
    probs = collect(probs)
    names = collect(names)
    if !isnothing(actual_country)
        println(title, " - Index: ", findfirst(==(actual_country), names), " Country: ", actual_country)
    end
    return Plots.bar(names[1:bars], probs[1:bars], legend=false, xlabel="Countries", xrotation=15, ylabel="Probabilities", title=title, color = ifelse.(names[1:bars] .== actual_country, :red, :blue))
end

# Additive / Laplace smoothing to avoid numerical issues with zero probabilities
function probability_smoothing(probs, epsilon = eps())
    probs .+= epsilon
    probs /= sum(probs)
    return probs
end

# Same as get_probability, but also taking in account the probability for neighboring states in the KNN for the continuous case
function node_probability_knn(node, marginal_state_dictionary, embedding, knn)
    prob = 0;
    actual_state = node.node_data["index"]
    for neighbor in knn[1][actual_state]
        marginal_state = marginal_state_dictionary[node]
        prob += get_probability(marginal_state, embedding, neighbor)
    end
    return prob
end

# Same as get_probability, but also taking in account the probability for neighboring states in the KNN for the discrete case
function node_probability_knn(node, marginal_state_dictionary, knn)
    prob = 0;
    actual_state = node.node_data["index"]
    for neighbor in knn[1][actual_state]
        prob += marginal_state_dictionary[node][1].state[neighbor]
    end
    return prob
end


# Helper functions for the result and analysis files

function format_result(input_array)
    return [row for row in eachrow(reduce(hcat, input_array))]
end

function parse_result(str)
    return parse.(Float64, split(replace(str, "Any" => "")[2:end-1], ", "))
end

function read_result(vector_str)
    result = []
    for line in vector_str
        push!(result, parse_result(line))
    end
    return format_result(result)
end
