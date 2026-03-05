using FileIO, Images
using TestImages
using ImageFiltering
using FFTW
using ImageQualityIndexes

#Process Photo:
gtImg = testimage("mandrill");
gtM = Float32.(Gray.(gtImg)) #this is photo's matrix in float (from [0:1])
r, c = size(gtM) #row and column

#Blur:
σ = 1.6 #st dv of blur in gaussian kernel
psf = Kernel.gaussian((σ, σ)) #gaussian Point Spread Function (PSF), PSF model give defocus blur from 
blurImg = imfilter(gtM, psf) #apply PSF filter to og image -> Blur = PSF * GT
noise = 0.15f0 #float, quite noisy
noise = noise .* randn(Float32,r, c) #randn is gaussian-shaped, array
blurImg =  clamp.(blurImg .+ noise, 0f0, 1f0) #clamp. (period for array) 2D array between 0-1
imshow(blurImg) # Blur = PSF * GT + noise

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
P = circshift(P, (-(kr ÷ 2), -(kc ÷ 2))) #this makes the cernal center be (1,1)
G = fft(blurImg) #frequency domain of blurred image
H = fft(P) #frequency response of blur kernel
K = 0.4f0 #Wiener stabilizer (smaller = sharper, more noise)
Hstar = conj.(H) #conjugate  
Habs = abs2.(H) #element-wise 
Fiener = (Hstar ./ (Habs .+ K)) .* G #Wiener filter
dbImg = real.(ifft(Fiener)) #real inverse of FFT
dbImg = clamp.(dbImg, 0f0, 1f0) #between 0-1
imshow(dbImg)

#Comparison:
psnr_bfr = ImageQualityIndexes.assess_psnr(blurImg, gtM) 
psnr_db = ImageQualityIndexes.assess_psnr(dbImg, gtM) 
ssim_db = ImageQualityIndexes.assess_ssim(dbImg, gtM)
println(psnr_bfr, psnr_db, ssim_db)

#save data
outdir = "~/results/wiener"
mkpath(outdir)
save(joinpath(outdir, "WdbImg.png"), colorview(Gray, dbImg)) #save db
open("results.txt", "a") do io
    println(io, psnr_bfr, " ", psnr_db, " ", ssim_db)
end 
#save comparison



