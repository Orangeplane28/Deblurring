include(joinpath(@__DIR__, "../../src/wiener.jl"))

using FileIO, Images
using DataFrames, CSV
using ImageQualityIndexes

const DATASET_DIR = raw"C:\Users\User\Downloads\dataset\dataset"

outdir = joinpath(@__DIR__, "../../results/outputs/wiener")
mkpath(outdir)
resultsf = joinpath(outdir, "nsr_experiment.csv")

image_names = ["1", "2", "3"]
nsr_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
results = []

println("Wiener NSR Experiment")
println()

for img_name in image_names

    println("Processing: $img_name")

    gt_loaded = load(joinpath(DATASET_DIR, "$img_name/groundtruth.tif"))
    blur_loaded = load(joinpath(DATASET_DIR, "$img_name/blurry.tif"))
    psf_loaded = load(joinpath(DATASET_DIR, "$img_name/kernel.tif"))

    gt = Float32.(channelview(gt_loaded))
    blurred = Float32.(channelview(blur_loaded))

    psf = Float32.(Gray.(psf_loaded))
    psf ./= sum(psf)

    best_psnr = -Inf
    best_ssim = -Inf
    best_nsr = 0.0
    best_restored = nothing

    for nsr in nsr_values
        println("    NSR = $nsr")
        restored = deblur(Wiener(Float32(nsr)), blurred, psf)

        current_psnr = assess_psnr(restored, gt)
        current_ssim = assess_ssim(restored, gt)

        println("        PSNR: $(round(current_psnr, digits=3))")
        println("        SSIM: $(round(current_ssim, digits=3))")
        if current_psnr > best_psnr
            best_psnr = current_psnr
            best_ssim = current_ssim
            best_nsr = nsr
            best_restored = restored
        end
    end

    save(joinpath(outdir, "$(img_name)_best.png"), colorview(RGB, best_restored))

    push!(results,
        (
            image = img_name,
            best_nsr = best_nsr,
            best_psnr = best_psnr,
            best_ssim = best_ssim
        )
    )
    println()
end

df = DataFrame(results)
CSV.write(resultsf, df)

println("DONE")
println(df)
