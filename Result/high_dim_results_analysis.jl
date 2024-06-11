include("../utils/load.jl")

#=
    File to analyze the result file generated by high_dim_results.jl.
=#

filepath = "Result/Files/high_dim_results_flights_master.csv"

data = DataFrame(CSV.File(filepath))
embedding_dims = [Integer(x) for x in parse_result(data.DIMS[1])]

landmark_diff_times = read_result(data.LANDMARK_DIFFUSION_TIMES)
landmark_inference_times = read_result(data.LANDMARK_INFERENCE_TIMES)
for (index, embedding_dim) in enumerate(embedding_dims)
    println("Landmark diffusion times dim=", embedding_dim, " : mean ",  mean(landmark_diff_times[index]), " var ", var(landmark_diff_times[index]), " median ", median(landmark_diff_times[index]))
end
println()
for (index, embedding_dim) in enumerate(embedding_dims)
    println("Landmark inference times dim=", embedding_dim, " : mean ",  mean(landmark_inference_times[index]), " var ", var(landmark_inference_times[index]), " median ", median(landmark_inference_times[index]))
end

println()
println("Total migrations: mean ", mean(data.TOTAL_MIGRATIONS), " var ", var(data.TOTAL_MIGRATIONS))
println("Average migrations: mean ", mean(data.AVERAGE_MIGRATIONS), " var ", var(data.AVERAGE_MIGRATIONS))
println("Minimum migrations: mean ", mean(data.MINIMUM_MIGRATIONS), " var ", var(data.MINIMUM_MIGRATIONS))
println("Total branchlength: mean ", mean(data.TOTAL_BRANCHLENGTHS), " var ", var(data.TOTAL_BRANCHLENGTHS))
println("Minimum migration over all runs:", minimum(data.TOTAL_MIGRATIONS), " - maximum: ", maximum(data.TOTAL_MIGRATIONS))
println()
