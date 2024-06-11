# File to load all the required packages and files for the project
using MolecularEvolution, CSV, LinearAlgebra, StatsBase, Distributions, NearestNeighbors, DataFrames, SparseArrays, Distances, MultivariateStats, Plots, Phylo, Random, Compose, Colors, Dates, StatsPlots

include("continuous_model_setup.jl")
include("model_and_partition.jl")
include("model_eval.jl")
include("MolecularEvolution_functions.jl")
