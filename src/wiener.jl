using FileIO, Images
using ImageFiltering
using FFTW
using ImageQualityIndexes

#wiener deconvolution function
function wiener_deconv(blurImg_ch::Matrix{Float32}, psf, K::Float32)

    r, c = size(blurImg_ch)

    psfM = Float32.(collect(psf))
    kr, kc = size(psfM)
    P = zeros(Float32, r, c)
    P[1:kr, 1:kc] .= psfM
    P = circshift(P, (-(kr ÷ 2), -(kc ÷ 2)))

    H     = fft(P)
    Hstar = conj.(H)
    Habs  = abs2.(H)
    G     = fft(blurImg_ch)
    Fiener = (Hstar ./ (Habs .+ K)) .* G

    return clamp.(Float32.(real.(ifft(Fiener))), 0f0, 1f0)
end

#apply to all channels
function run_wiener_rgb(My_rgb::Array{Float32,3}, psf, K::Float32)

    _, r, c = size(My_rgb) 
    dbImg   = zeros(Float32, 3, r, c)

    for ch in 1:3
        dbImg[ch, :, :] = wiener_deconv(My_rgb[ch, :, :], psf, K)
    end

    return dbImg, psf
end


#main:

#psf assumption
#σ   = 1.6
#psf = Kernel.gaussian((σ, σ))

#deblurring
#K = 0.3f0
