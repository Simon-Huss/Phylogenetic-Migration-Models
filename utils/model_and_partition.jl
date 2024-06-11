#=
    Function to simulate a continuous-time Markov chain
    D: Data to generate or calculate the transition rate matrix Q (or the transition rate matrix itself)
    func: Function to calculate the jump matrix (row) using the data D and the current state, it must be callable like func(D, state; kwargs) where state is an integer
    counter: Counter to keep track of the number of migrations that occur
    
    Example initialization if the data D is a Q-matrix and Q_to_J being a function that extracts the row X from the corresponding jump matrix:
        model = Gillespie_CTMC(Q, (Q, X) -> Q_to_J(Q[X, :], X))

=#

# Helper struct to keep track of the number of migrations in the CTMC
mutable struct Counter
    value::Int
end

mutable struct Gillespie_CTMC <: DiscreteStateModel
    D
    func::Function 
    counter::Counter
    function Gillespie_CTMC(D, func::Function)
        new(D, func, Counter(0))
    end
end

# Partition that keeps track of the current state
mutable struct Partition_CTMC <: Partition
    state
    function Partition_CTMC(state)
        new(state)
    end 
end

# Function to simulate the continuous-time Markov chain and also calculate the amount of migrations that occur
function MolecularEvolution.forward!(
    dest::Partition_CTMC,
    source::Partition_CTMC,
    model::Gillespie_CTMC,
    node::FelNode
)   
    current_migrations = model.counter.value
    dest.state = simulate_CTMC(model.D, model.func, source.state, node.branchlength, counter=model.counter)
    node.node_data["migrations"] = model.counter.value - current_migrations
    if MolecularEvolution.isroot(node)
        node.node_data["migrations_from_root"] = 0
    else
        node.node_data["migrations_from_root"] = node.node_data["migrations"] + node.parent.node_data["migrations_from_root"]
    end
end

# Not applicable
function MolecularEvolution.sample_partition!(Partition::Partition_CTMC)
end
