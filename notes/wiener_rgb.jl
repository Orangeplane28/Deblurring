using FileIO, Images
using ImageFiltering
using TestImages
using FFTW
using ImageQualityIndexes

#Process Photo:
gtImg = load("C:/Users/User/Downloads/Deblurring/data/train/sharp/0.png")
gtM = Float32.(channelview(RGB.(gtImg))) # (3, H, W) array for RGB
_, r, c = size(gtM) # channel count, row, column

#Blur:
σ = 1.6 #st dv of blur in gaussian kernel
psf = Kernel.gaussian((σ, σ)) #gaussian Point Spread Function (PSF), PSF model give defocus blur
blurImg = zeros(Float32, 3, r, c) #apply PSF filter to each RGB channel
for ch in 1:3
    blurImg[ch, :, :] = imfilter(gtM[ch, :, :], psf) # Blur = PSF * GT for each channel
end
noise = 0.05f0 #float, quite noisy
noise = noise .* randn(Float32, 3, r, c) #randn is gaussian-shaped, 3D array for RGB
blurImg = clamp.(blurImg .+ noise, 0f0, 1f0) #clamp 3D array between 0-1
# Blur = PSF * GT + noise

#Deblur:
#Wiener Deconvolution (stabilizes inversion)
# blurImg = PSF * gtM + noise
# Fourier domain: fft(blurImg) = fft(PSF) * fft(gtM) + noise  -> waves
# Wiener filter:  F̂ = (fft(PSF)* / (|fft(PSF)|^2 + K)) .* fft(blurImg)  -> wiener estimate
# f = ifft(F̂) -> inverse it

psfM = Float32.(collect(psf)) #make array of PSF
kr, kc = size(psfM)

P = zeros(Float32, r, c) #make psf as big as image
P[1:kr, 1:kc] .= psfM #place psfM
P = circshift(P, (-(kr ÷ 2), -(kc ÷ 2))) #this makes the kernel center be (1,1)
K = 0.3f0 #Wiener stabilizer (smaller = sharper, more noise)
H = fft(P) #frequency response of blur kernel
Hstar = conj.(H) #conjugate
Habs = abs2.(H) #element-wise

dbImg = zeros(Float32, 3, r, c) #deblurred image for each RGB channel
for ch in 1:3
    G = fft(blurImg[ch, :, :]) #frequency domain of blurred channel
    Fiener = (Hstar ./ (Habs .+ K)) .* G #Wiener filter
    dbImg[ch, :, :] = real.(ifft(Fiener)) #real inverse of FFT for this channel
end
dbImg = clamp.(dbImg, 0f0, 1f0) #between 0-1

#Comparison:
psnr_bfr = ImageQualityIndexes.assess_psnr(blurImg, gtM)
psnr_db = ImageQualityIndexes.assess_psnr(dbImg, gtM)
ssim_db = ImageQualityIndexes.assess_ssim(dbImg, gtM)
println(psnr_bfr, psnr_db, ssim_db)

#save data
outdir = "C:/Users/User/Downloads/Deblurring/results"
mkpath(outdir)
save(joinpath(outdir, "dbImg0.png"), colorview(RGB, dbImg)) #RGB db
save(joinpath(outdir, "bImg0.png"), colorview(RGB, blurImg)) #RGB y

open(joinpath(outdir, "results.txt"), "a") do io
    println(io, psnr_bfr, " ", psnr_db, " ", ssim_db)
end
