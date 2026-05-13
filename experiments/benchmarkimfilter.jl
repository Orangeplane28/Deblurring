include(joinpath(@__DIR__, "../src/PAMde.jl"))

using ImageFiltering, ImageFiltering.Models, LinearAlgebra
using FileIO, Images, OffsetArrays
using BenchmarkTools
using DataFrames, CSV
using Plots, StatsPlots

#Make outputs 
outdir = joinpath(@__DIR__, "../results/outputs/benchmark")
mkpath(outdir)
resultsf = joinpath(outdir, "benchmark_imfilter.csv")

#Load image from test data
img_loaded = load(joinpath(@__DIR__, "../data/test/blur/33.png"))
img_rgb = Float64.(channelview(img_loaded))

image_sizes = [360, 640, 1024, 1280]
kernel_sizes = [11, 15, 17, 21, 23]
results = []

println("IMFILTER Benchmark")
println()

for img_size in image_sizes
    println("Image size: $img_size")
    
    for k_size in kernel_sizes
        #resize
        resized_rgb = zeros(Float64, 3, img_size, img_size)
        for ch in 1:3
            resized_rgb[ch, :, :] = imresize(img_rgb[ch, :, :], (img_size, img_size))
        end

        test_image = resized_rgb[1, :, :]
        test_kernel = fill(1.0 / k_size^2, k_size, k_size)
        ph, pw = (k_size - 1) ÷ 2, (k_size - 1) ÷ 2
        padded_image = OffsetArrays.no_offset_view(collect(padarray(test_image, Pad(:replicate, ph, pw))))
        
        #Benchmark imfilter
        bench = @benchmark imfilter($padded_image, centered(reverse($test_kernel, dims=(1,2))), Inner())
        
        time_ms = mean(bench.times) / 1e6 # median instead?
        tallocs = bench.allocs
        
        println("Kernel $k_size| Time $(round(time_ms, digits=4)) ms| Alloc: $tallocs")
        
        push!(results, (image_size=img_size, kernel_size=k_size, time_ms=time_ms, allocs=tallocs))
    end
    println()
end

df = DataFrame(results)
CSV.write(resultsf, df)
open(joinpath(outdir, "benchmark_imfilter.txt"), "w") do f #write entire dataframe
    show(f, df)
end
println("DONE")
println(df)

#plotB = plot(title="imfilter Benchmark", xlabel="Image Size (px)", ylabel="Time (ms)")
#plot!
df.kcolumn = string.(df.kernel_size)
@df df plot(:image_size, :time_ms, group=:kcolumn, marker=:cross, xlabel="Image Size (px)", ylabel="Time (ms)", title="imfilter Benchmark")
savefig("results/outputs/benchmark/imfilter_plot.png") #save last plot that was created