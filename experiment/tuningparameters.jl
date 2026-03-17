include(joinpath(@__DIR__, "../src/PAMde.jl"))
include(joinpath(@__DIR__, "../src/WEINERde.jl"))

using ImageFiltering, ImageFiltering.Models, LinearAlgebra
using FileIO, Images, OffsetArrays
using ImageView
using Plots
using Dates

#make outputs 
outdir = joinpath(@__DIR__, "../results/outputs")
mkpath(outdir)
resultsf = joinpath(outdir, "results.txt")

#Load images
#PAM
gtImgB = load(joinpath(@__DIR__, "../data/blurred.jpg"))
gtImgGT = load(joinpath(@__DIR__, "../data/groundtruth.jpg"))
My_rgb = Float64.(channelview(gtImgB))
M_GT =  Float64.(channelview(gtImgGT))
#WIENER
My_rgbW = Float32.(My_rgb)


# Initial kernel
k_size = 11
k_init = fill(1.0 / k_size^2, k_size, k_size)

#PAM parameters
λ0 = 1.5
λmin = 0.0006
ϵ_x = 0.05
ϵ_k = 1e-6
stop = 0.01
max_coarse = 300
max_fine = 800
#WIENER
σ_blur = 1.6 #gaussian approximation
psf = Kernel.gaussian((σ_blur))
K_wiener = Float32(0.3f0)



#PAM with Pyramich scheme
t1 = time()
x_p, k_p = run_pyramid_rgb(My_rgb, k_init, λ0, λmin, ϵ_x, ϵ_k, stop, max_coarse, max_fine)
time_p = time() - t1
println("PAM Pyramid done in ", time_p)

#Wiener Decon
t2 = time()
x_w, _ = run_wiener_rgb(My_rgbW, psf, Float32(K_wiener))
time_w = time() - t2
println("Wiener done in", time_w)
x_w = convert(Array{Float64,3}, x_w)
#PAM 
t3 = time()
x_pam, k_pam = run_pam_rgb(My_rgb, k_init, λ0, λmin, ϵ_x, ϵ_k, stop, 1000)
time_pam = time() - t3
println("PAM done in", time_pam)
#make it cropped
_, r, c = size(My_rgb)
h_pam = (k_size - 1) ÷ 2
w_pam = (k_size - 1) ÷ 2
x_pam = x_pam[:, 1+h_pam:r+h_pam, 1+w_pam:c+w_pam]

println("Now Metrics")
#PSNR and SSIM
psnr_pyramid = assess_psnr(x_p, M_GT)
ssim_pyramid = assess_ssim(x_p, M_GT)

psnr_wiener = assess_psnr(x_w, M_GT)
ssim_wiener = assess_ssim(x_w, M_GT)

psnr_pam = assess_psnr(x_pam, M_GT)
ssim_pam = assess_ssim(x_pam, M_GT)

#also blurred input:
psnr_blurred = assess_psnr(My_rgb, M_GT)

#save deblurred
save(joinpath(outdir, "deblurred_pyramid.png"), colorview(RGB, x_p))
save(joinpath(outdir, "deblurred_wiener.png"), colorview(RGB, x_w))
save(joinpath(outdir, "deblurred_pam.png"), colorview(RGB, x_pam))

#save
println("Now File")
open(resultsf, "a") do io
    println(io, "Pyramid:")
    println(io, "PSNR: ", psnr_pyramid)
    println(io, "SSIM:", ssim_pyramid)

    println(io, "PAM:")
    println(io, "PSNR: ", psnr_pam)
    println(io, "SSIM:", ssim_pam)

    println(io, "Wiener:")
    println(io, "PSNR: ", psnr_wiener)
    println(io, "SSIM:", ssim_wiener)

end




#PAM parameters
#λ0 = 1.5
#λmin = 0.0006
#ϵ_x = 0.05
#ϵ_k = 1e-6
#stop = 0.01
#max_coarse = 300
#max_fine = 800
#WIENER
#σ_blur = 1.6 #gaussian approximation
#psf = Kernel.gaussian((σ_blur))
#K_wiener = Float32(0.3f0)



# Initial kernel
#k_size = 15
#k_init = fill(1.0 / k_size^2, k_size, k_size)

#PAM parameters
#λ0 = 2.0
#λmin = 0.0006
#ϵ_x = 0.009
#ϵ_k = 1e-6
#stop = 0.01
#max_coarse = 350
#max_fine = 1000
#WIENER
#σ_blur = 1.6 #gaussian approximation
#psf = Kernel.gaussian((σ_blur))
#K_wiener = Float32(0.6f0)

