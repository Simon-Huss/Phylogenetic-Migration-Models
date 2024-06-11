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


# Creates a dictionary of all the child counts (including the node itself) which can then be used by ladderize to sort the nodes
function countchildren(tree::AbstractTreeNode)
    nodes = MolecularEvolution.getleaflist(tree)
    child_count = Dict{FelNode, Int}()
    skipped_nodes = []
    checked_nodes = Set{FelNode}()

    while !isempty(nodes)
        node = pop!(nodes)

        # If the node does not have children, check if the parent has been checked, if not, push the parent to the stack
        if isempty(node.children)
            child_count[node] = 1
            push!(checked_nodes, node)
            if !isnothing(node.parent) && !in(node.parent, checked_nodes)
                push!(nodes, node.parent)
            end
        # If the node has children, check if all children have been checked, if not, push the current node to the skipped stack for later
        else
            all_children_checked = true
            for child in node.children
                # If a child has not been checked, we push the current node to the skipped stack for later and continue popping from nodes
                if !in(child, checked_nodes)
                    all_children_checked = false
                    push!(skipped_nodes, node)
                    break
                end
            end
            # If all children have been checked, we count all the children and mark the current node as checked
            if all_children_checked
                child_count[node] = 1 + sum([child_count[child] for child in node.children])
                push!(checked_nodes, node)
                if !isnothing(node.parent) && !in(node.parent, checked_nodes)
                    push!(nodes, node.parent)
                end
            end
            
        end
        # When the current node stack is empty, we check if there are any skipped nodes and if so, check those as well and repeat until all nodes have been checked
        if isempty(nodes) && !isempty(skipped_nodes)
            nodes = skipped_nodes
            skipped_nodes = []
        end
    end
    return child_count
end

ladderize!
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
