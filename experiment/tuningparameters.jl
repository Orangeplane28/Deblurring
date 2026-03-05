include(joinpath(@__DIR__, "../src/PAMde.jl"))
using ImageFiltering, ImageFiltering.Models, LinearAlgebra
using FileIO, Images, OffsetArrays
using ImageView
using Plots

# Load image
gtImg = load("C:/Users/User/Downloads/Deblurring/data/blurred.jpg")
My_rgb = Float64.(channelview(gtImg))

# Initial kernel
k_size = 11
k_init = fill(1.0 / k_size^2, k_size, k_size)
λmin   = 0.0006
max_steps = 1000

results = []

t = time()
x, k = run_pyramid_rgb(My_rgb, k_init, 1.5, 0.0006, 0.05, 1e-6, 0.01, 300, 800)
println("Single run: $(round(time()-t, digits=1))s")
imshow(colorview(RGB, x))

heatmap(k, title="kernel")
savefig("kernel.png")