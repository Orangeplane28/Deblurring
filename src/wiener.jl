using FFTW

"""
    wiener_deblur(My_rgb, k, nsr)

Apply Wiener non-blind deconvolution to a RGB image using FFT-domain inversion.

# Arguments
- My_rgb: blurred RGB image as `AbstractArray{<:Real,3}` (3×H×W)
- k: blur kernel (psf)
- nsr: noise-to-signal ratio regularization parameter

# Returns
- Returns deblurred image
"""

###################################
# Deblurring Interface
###################################

abstract type DeblurMethod end
struct Wiener <: DeblurMethod
    nsr::Float32
end

function deblur(
    method::Wiener,
    My_rgb::AbstractArray{<:Real,3},
    k
)
    return _wiener_rgb(Float32.(My_rgb), k, method.nsr)
end

###################################
# Wiener RGB Deblurring
###################################

function _wiener_rgb(My_rgb::Array{Float32,3}, k, nsr::Float32)

    nsr > 0 || throw(ArgumentError("nsr must be positive"))

    _, r, c = size(My_rgb)
    nsr = Float32(nsr)

    kM = Float32.(collect(k))
    kr, kc = size(kM)

    @assert isodd(kr) && isodd(kc) "kernel dimensions must be odd"
    kr ≤ r && kc ≤ c || throw(ArgumentError("kernel ($kr×$kc) larger than image ($r×$c)"))

    #padded/center kernel:
    kPad = zeros(Float32, r, c)
    kPad[1:kr, 1:kc] .= kM
    kPad = circshift(kPad, (-(kr ÷ 2), -(kc ÷ 2)))


    #Precompute K for Wiener inversion
    Kfft  = fft(kPad)
    Kstar = conj.(Kfft)
    Kabs  = abs2.(Kfft)

    dbImg = zeros(Float32, 3, r, c)
    for ch in 1:3
        dbImg[ch, :, :] = _wiener_channel(My_rgb[ch, :, :], Kstar, Kabs, nsr)
    end

    return dbImg
end


###################################
# Single-channel FFT
###################################

function _wiener_channel(
    blurImg_ch::Matrix{Float32}, 
    Kstar, 
    Kabs, 
    nsr::Float32
    )

    Y = fft(blurImg_ch)
    #Wiener Inverse Filter:
    X = (Kstar ./ (Kabs .+ nsr)) .* Y
    return clamp.(real.(ifft(X)), 0f0, 1f0)
end

