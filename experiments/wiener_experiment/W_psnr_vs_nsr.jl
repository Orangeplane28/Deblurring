#Experiment: NSR's effect on Wiener deconvolution
#
#Design:
#-Treatment factor: NSR parameter
#-Blocking factor: images
#-Experimental unit: one blurred image reconstructed with Wiener under one NSR
#-Response variables: PSNR, SSIM, ΔPSNR, ΔSSIM
#-Control variables: images, PSF, dimensions, color representation, deblurring

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
include(joinpath(@__DIR__, "../../src/wiener.jl"))

using FileIO, Images
using DataFrames, CSV
using ImageQualityIndexes
using Statistics
using GLM
using StatsModels

const DATASET_DIR = raw"C:\Users\User\Downloads\dataset\dataset"
outdir = joinpath(@__DIR__, "../../results/outputs/wiener")
mkpath(outdir)
raw_results_file = joinpath(outdir, "wiener_nsr_full_results.csv")

image_names = ["1", "2", "3"]
nsr_values = Float32.(10.0 .^ (-6:0.25:-2))
results = []

# ======================================
# Data Collection
# ======================================

println("Wiener NSR Experiment")
for img_name in image_names
    println("Image: $img_name")
    gt_loaded = load(joinpath(DATASET_DIR, "$img_name/groundtruth.tif"))
    blur_loaded = load(joinpath(DATASET_DIR, "$img_name/blurry.tif"))
    psf_loaded = load(joinpath(DATASET_DIR, "$img_name/kernel.tif"))

    gt = Float32.(channelview(float.(RGB.(gt_loaded))))
    blurred = Float32.(channelview(float.(RGB.(blur_loaded))))
    psf = Float32.(Gray.(psf_loaded))
    psf ./= sum(psf)

    baseline_psnr = assess_psnr(blurred, gt)
    baseline_ssim = assess_ssim(blurred, gt)

    best_diff_psnr  = -Inf
    worst_diff_psnr = Inf
    best_restored   = nothing
    worst_restored  = nothing

    for nsr in nsr_values
        println("NSR = $nsr")
        restored   = deblur(Wiener(nsr), blurred, psf)
        psnr_value = Float64(assess_psnr(restored, gt))
        ssim_value = Float64(assess_ssim(restored, gt))

        diff_psnr = psnr_value - Float64(baseline_psnr)
        diff_ssim = ssim_value - Float64(baseline_ssim)
        log_nsr   = log10(Float64(nsr))
        push!(results,
            (
                image = img_name,
                nsr = Float64(nsr),
                log_nsr = log_nsr,
                log_nsr_sq = log_nsr^2,
                psnr = psnr_value,
                ssim = ssim_value,
                baseline_psnr = Float64(baseline_psnr),
                baseline_ssim = Float64(baseline_ssim),
                diff_psnr = diff_psnr,
                diff_ssim = diff_ssim,
            )
        )
        if diff_psnr > best_diff_psnr
            best_diff_psnr = diff_psnr
            best_restored  = copy(restored)
        end
        if diff_psnr < worst_diff_psnr
            worst_diff_psnr = diff_psnr
            worst_restored  = copy(restored)
        end
    end

    best_comparison = hcat(colorview(RGB, gt), colorview(RGB, blurred), colorview(RGB, best_restored))
    worst_comparison = hcat(colorview(RGB, gt), colorview(RGB, blurred), colorview(RGB, worst_restored))
    save(joinpath(outdir, "$(img_name)_BEST.png"),  clamp01.(best_comparison))
    save(joinpath(outdir, "$(img_name)_WORST.png"), clamp01.(worst_comparison))

    println()
end

# ======================================
# Dataframe
# ======================================

df = DataFrame(results)
sort!(df, [:image, :diff_psnr]; rev = [false, true])
CSV.write(raw_results_file, df)

println("Data Frame:")
println(df)


# ======================================
# Quadratic Regression
# ======================================

println()
println("Quadratic Regression")
regression_results = []

for image_df in groupby(df, :image)

    img_name = first(image_df.image)
    model = lm(
        @formula(diff_psnr ~ log_nsr + log_nsr_sq),
        image_df
    )

    β = coef(model)
    β2 = β[3]
    corr_r2 = r2(model)

    println()
    println("Image: $img_name")
    push!(
        regression_results,
        (
            image = img_name,
            r2 = corr_r2,
            downwards = β2 < 0
        )
    )
end

regression_df = DataFrame(regression_results)
println()
println("Regression Summary:")
println(regression_df)

CSV.write(
    joinpath(outdir, "quadratic_regression_results.csv"),
    regression_df
)
