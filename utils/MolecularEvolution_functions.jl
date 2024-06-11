#=
    This file contains some helper functions for the MolecularEvolution.jl framework, for future reference these might already be implemented in the package.
    Mainly the functions have been changed to be non-recursive to avoid stack overflow errors when working with large trees as well as some custom partition types for lower memory usage.
=#
using MolecularEvolution

function MolecularEvolution.getleaflist(node::T) where {T<:AbstractTreeNode}
    leaflist = []
    nodes = [node]
    while nodes != []
        node = pop!(nodes)
        if node.children == []
            push!(leaflist, node)
        else 
            for child in node.children
                push!(nodes, child)
            end
        end
    end
    return leaflist
end

function MolecularEvolution.getnonleaflist(node::T) where {T<:AbstractTreeNode}
    nonleaflist = []
    nodes = [node]
    while nodes != []
        node = pop!(nodes)
        if node.children != []
            push!(nonleaflist, node)
            for child in node.children
                push!(nodes, child)
            end
        end
    end
    return nonleaflist
end


function MolecularEvolution.getnodelist(node::T) where {T<:AbstractTreeNode}
    nodelist = []
    nodes = [node]
    while nodes != []
        node = pop!(nodes)
        push!(nodelist, node)
        for child in node.children
            push!(nodes, child)
        end
    end
    return nodelist
end

function MolecularEvolution.sample_down!(node::FelNode, models, partition_list)
    stack = [node]
    while !isempty(stack)
        node = pop!(stack)
        model_list = models(node)
        for part in partition_list
            if MolecularEvolution.isroot(node)
                forward!(node.message[part], node.parent_message[part], model_list[part], node)
            else
                forward!(node.message[part], node.parent.message[part], model_list[part], node)
            end
            sample_partition!(node.message[part])
        end
        if !isleafnode(node)
            for child in reverse(node.children)
                push!(stack, child)
            end
        end
    end
end

# Dynamic programming approach to ladderize by first counting all the children of each node and then sorting
function MolecularEvolution.ladderize!(tree::T) where {T<:AbstractTreeNode}
    child_counts = countchildren(tree)
    for node in MolecularEvolution.getnodelist(tree)
        if !isempty(node.children)
            sort!(node.children, lt = (x, y) -> child_counts[x] < child_counts[y])
        end
    end
end


# Creates a dictionary of all the child counts which can then be used by ladderize to sort the nodes
function countchildren(tree::AbstractTreeNode)
     # Initialize the dictionary to store the number of children for each node
     children_count = Dict{FelNode, Int}()

     # Initialize the stack for DFS
     stack = [tree]
     
     # Initialize a list to keep track of the post-order traversal
     post_order = []
 
     # First pass: Perform DFS and store the nodes in post-order
     while !isempty(stack)
        node = pop!(stack)
        push!(post_order, node)
        for child in node.children
            push!(stack, child)
        end
     end
 
     # Second pass: Calculate the number of children for each node in post-order
     for node in reverse(post_order)
        count = 0
        for child in node.children
            count += 1 + children_count[child]
        end
        children_count[node] = count
     end
 
    return children_count
end

# Same as the one implemented in MolecularEvolution, but it prefers the left interval if both are mutually beneficial
function MolecularEvolution.golden_section_maximize(f, a::Real, b::Real, transform, tol::Real)
    a, b = min(a, b), max(a, b)
    h = b - a
    if h <= tol
        return transform((a + b) / 2)#(a,b)
    end
    # required steps to achieve tolerance
    n = Int(ceil(log(tol / h) / log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(transform(c))
    yd = f(transform(d))
    for k = 1:(n-1)
        if yd > yc
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(transform(d))           
        else
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(transform(c))
        end
    end
    if yd > yc
        return transform((c + b) / 2)#(c,b)
    else
        return transform((a + d) / 2)#(a,d)
    end
end

const invphi = 1 / MathConstants.φ
const invphi2 = 1 / MathConstants.φ^2


# Custom Discrete Partition types with lower memory usage than the default Float64 implementation
mutable struct CustomDiscretePartitionFloat16 <: DiscretePartition
    state::Array{Float16,2}
    states::Int
    sites::Int
    scaling::Array{Float16,1}
    function CustomDiscretePartitionFloat16(states, sites)
        new(zeros(states, sites), states, sites, zeros(sites))
    end
    function CustomDiscretePartitionFloat16(freq_vec::Vector{Float16}, sites::Int8) #Add this constructor to all partition types
        state_arr = zeros(length(freq_vec), sites)
        state_arr .= freq_vec
        new(state_arr, length(freq_vec), sites, zeros(sites))
    end
    function CustomDiscretePartitionFloat16(state, states, sites, scaling)
        @assert size(state) == (states, sites)
        new(state, states, sites, scaling)
    end
end

mutable struct CustomDiscretePartitionFloat32 <: DiscretePartition
    state::Array{Float32,2}
    states::Int
    sites::Int
    scaling::Array{Float32,1}
    function CustomDiscretePartitionFloat32(states, sites)
        new(zeros(states, sites), states, sites, zeros(sites))
    end
    function CustomDiscretePartitionFloat32(freq_vec::Vector{Float32}, sites::Int8) #Add this constructor to all partition types
        state_arr = zeros(length(freq_vec), sites)
        state_arr .= freq_vec
        new(state_arr, length(freq_vec), sites, zeros(sites))
    end
    function CustomDiscretePartitionFloat32(state, states, sites, scaling)
        @assert size(state) == (states, sites)
        new(state, states, sites, scaling)
    end
end
