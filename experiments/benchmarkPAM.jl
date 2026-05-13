include(joinpath(@__DIR__, "../src/PAMde.jl"))

using ImageFiltering, ImageFiltering.Models, LinearAlgebra
using FileIO, Images, OffsetArrays
using BenchmarkTools
using DataFrames, CSV, Plots, StatsPlots

#Make outputs 
outdir = joinpath(@__DIR__, "../results/outputs/benchmark")
mkpath(outdir)
resultsf = joinpath(outdir, "benchmark_pam_methods.csv")

#Load image from test data
img_loaded = load(joinpath(@__DIR__, "../data/test/blur/33.png"))
img_rgb = Float64.(channelview(img_loaded))
img_in = size(img_rgb, 2)  #dimensions

#Test sizes
image_sizes = [360, 640, 1024]
kernel_sizes = [11, 15, 23]

#PAM parameters
λ0 = 1.5
λmin = 0.0006
ϵ_x_in = 0.05
ϵ_k_in = 1e-6 #has to change inversely to image resolution
stop = 0.01
max_iters = 200
max_coarse = 100
max_fine = 150

results = []

println("PAM Benchmark")
println()
for img_size in image_sizes
    for k_size in kernel_sizes
        println("Image size: $img_size, Kernel: $k_size")
        ϵ_x = ϵ_x_in * (img_in / img_size)
        ϵ_k = ϵ_k_in * (img_in / img_size)

        #have to resize image
        resized_rgb = zeros(Float64, 3, img_size, img_size)
        for ch in 1:3
            resized_rgb[ch, :, :] = imresize(img_rgb[ch, :, :], (img_size, img_size))
        end
        resized_single = resized_rgb[1, :, :]
        
        #kernel
        k_init = fill(1.0 / k_size^2, k_size, k_size)
        
        #Benchmark 1: run_pam
        bench1 = @benchmark run_pam($resized_single, $k_init, $λ0, $λmin, $ϵ_x, $ϵ_k, $stop, $max_iters)
        time1_ms = mean(bench1.times) / 1e6 #measured in nanoseconds
        println("run_pam: $(round(time1_ms, digits=3)) ms")
        push!(results, (image_size=img_size, kernel_size=k_size, method="run_pam", time_ms=time1_ms, allocs=bench1.allocs)) #push results into array
        
        #Benchmark 2: run_pam_rgb
        bench2 = @benchmark run_pam_rgb($resized_rgb, $k_init, $λ0, $λmin, $ϵ_x, $ϵ_k, $stop, $max_iters)
        time2_ms = mean(bench2.times) / 1e6
        println("run_pam_rgb: $(round(time2_ms, digits=3)) ms")
        push!(results, (image_size=img_size, kernel_size=k_size, method="run_pam_rgb", time_ms=time2_ms, allocs=bench2.allocs))
        
        #Benchmark 3: run_pyramid_rgb
        bench3 = @benchmark run_pyramid_rgb($resized_rgb, $k_init, $λ0, $λmin, $ϵ_x, $ϵ_k, $stop, $max_coarse, $max_fine)
        time3_ms = mean(bench3.times) / 1e6
        println("run_pyramid_rgb: $(round(time3_ms, digits=3)) ms")
        push!(results, (image_size=img_size, kernel_size=k_size, method="run_pyramid_rgb", time_ms=time3_ms, allocs=bench3.allocs))
        
        println()
    end
end

df = DataFrame(results)
CSV.write(resultsf, df)
open(joinpath(outdir, "benchmark_imfilter.txt"), "w") do f
    show(f, df)
end

println("DONE")
println(df)
df.kcolumn = string.(df.kernel_size)

#seperate dataframe for each method with statsplot
df1 = filter(r -> r.method == "run_pam", df)
@df df1 plot(:image_size, :time_ms, group=:kcolumn, marker=:circle, xlabel="Image Size (px)", ylabel="Time (ms)", title="Run PAM ")
savefig(joinpath(outdir, "plot_run_pam.png"))

df2 = filter(r -> r.method == "run_pam_rgb", df)
@df df2 plot(:image_size, :time_ms, group=:kcolumn, marker=:circle, xlabel="Image Size (px)", ylabel="Time (ms)", title="Run PAM RGB")
savefig(joinpath(outdir, "plot_run_pam_rgb.png"))

df3 = filter(r -> r.method == "run_pyramid_rgb", df)
@df df3 plot(:image_size, :time_ms, group=:kcolumn, marker=:circle, xlabel="Image Size (px)", ylabel="Time (ms)", title="Run PAM Pyramid ")
savefig(joinpath(outdir, "plot_run_pyramid_rgb.png"))