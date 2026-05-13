module Deblurring

include("pam.jl")
include("wiener.jl")
include("hyper_laplacian.jl")

export pam_deblur
export wiener_deblur
export hyper_deblur

end