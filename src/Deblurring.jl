module Deblurring

include("pam.jl")
include("wiener.jl")
include("hyper_laplacian.jl")

export DeblurMethod
export PAM
export Wiener
export HyperLaplacian
export deblur

end