include(joinpath(@__DIR__, "../src/PAMde.jl"))
include(joinpath(@__DIR__, "../src/hyperlaplaciande.jl"))

using ImageFiltering, ImageFiltering.Models, LinearAlgebra
using FileIO, Images, OffsetArrays
using FFTW
using Plots
using Dates
 
FFTW.set_num_threads(7)
 
#outputs
outdir = joinpath(@__DIR__, "../results/outputs")
mkpath(outdir)
resultsf = joinpath(outdir, "results_pyramid.txt")
 
#images
gtImgB  = load(joinpath(@__DIR__, "../data/test/blur/33.png"))
gtImgGT = load(joinpath(@__DIR__, "../data/test/sharp/33.png"))
My_rgb  = Float64.(channelview(gtImgB))
M_GT    = Float64.(channelview(gtImgGT))
 
k_size = 23
k_init = fill(1.0 / k_size^2, k_size, k_size)
λ0 = 1.5
λmin = 0.0006
ϵ_x = 0.015
ϵ_k = 1e-6
stop = 0.01
max_coarse = 300
max_fine = 1000
 
#run
t1 = time()
alloc_bytes = @allocated x_p, k_p = run_pyramid_rgb(My_rgb, k_init, λ0, λmin, ϵ_x, ϵ_k, stop, max_coarse, max_fine)
time_p = time() - t1
println("Pyramid done in $(round(time_p, digits=1))s  allocs=$(round(alloc_bytes/1e6, digits=1))MB")
 
#metrics
psnr_blurred = assess_psnr(My_rgb, M_GT)
psnr_pyramid = assess_psnr(x_p, M_GT)
ssim_pyramid = assess_ssim(x_p, M_GT)
println("Blurred  PSNR=$(round(psnr_blurred, digits=3))")
println("Pyramid  PSNR=$(round(psnr_pyramid, digits=3))  SSIM=$(round(ssim_pyramid, digits=4))")
 
#save deblurred image
save(joinpath(outdir, "deblurred_pyramid_op.png"), colorview(RGB, x_p))
 
#kernel heatmap
heatmap(k_p, color=:hot, title="Estimated kernel ($(k_size)×$(k_size))", aspect_ratio=:equal, axis=false, colorbar=true)
savefig(joinpath(outdir, "kernel_heatmap_op.png"))

#kernel refinement
k_refined = copy(k_p)
k_refined[k_refined .< 0.1 * maximum(k_p)] .= 0.0
k_refined ./= sum(k_refined)
 
#hyper-laplacian non-blind deconvolution
λ_hl = 100.0
 
t2 = time()
x_hl = hyperlaplacian_deconv_rgb(My_rgb, k_refined, λ_hl)
time_hl = time() - t2
println("Hyper-Laplacian done in $(round(time_hl, digits=2))s")
 
psnr_hl = assess_psnr(x_hl, M_GT)
ssim_hl = assess_ssim(x_hl, M_GT)
println("HyperLap PSNR=$(round(psnr_hl, digits=3))  SSIM=$(round(ssim_hl, digits=4))")
save(joinpath(outdir, "deblurred_hyperlaplacian.png"), colorview(RGB, x_hl))
 
#results file
open(resultsf, "a") do io
    println(io, "=== $(Dates.now()) ===")
    println(io, "k=$(k_size)  λ0=$(λ0)  λmin=$(λmin)  ϵ_x=$(ϵ_x)  ϵ_k=$(ϵ_k)  stop=$(stop)  coarse=$(max_coarse)  fine=$(max_fine)")
    println(io, "Time: $(round(time_p, digits=1))s  Allocs: $(round(alloc_bytes/1e6, digits=1))MB")
    println(io, "Blurred PSNR: $(psnr_blurred)")
    println(io, "Pyramid PSNR: $(psnr_pyramid)  SSIM: $(ssim_pyramid)")
    println(io, "k_max: $(round(maximum(k_p), digits=6))")
    println(io, "HLap Time: $(round(time_hl, digits=2))s")
    println(io, "HLap PSNR: $(psnr_hl)  SSIM: $(ssim_hl)")
    println(io, "")
end
