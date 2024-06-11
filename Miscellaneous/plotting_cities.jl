include("../utils/load.jl")
using PlotlyJS
# File to visualize the dataset
cities = 10_000

# 2-dimensions
N = get_coords_2D(cities, return_name=false)
plt2d = Plots.scatter(N[1, :], N[2, :], markersize=0.25, title="World cities", xlabel="Longitude", ylabel="Latitude", legend=false)
display(plt2d)

# 3-dimensions
M = get_coords(cities, return_name=false)
plt3d = PlotlyJS.scatter3d(x=M[1, :], y=M[2, :], z=M[3, :], mode="markers", marker=attr(size=1))
PlotlyJS.Plot(plt3d)