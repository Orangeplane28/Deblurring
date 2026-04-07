using ImageFiltering
using ImageFiltering.Models
using LinearAlgebra
using FileIO, Images
using OffsetArrays
using Images

mutable struct Deconvolution 
    x :: Matrix{Float64}
    k :: Matrix{Float64}
    λ::Float64 #weight for TV
    γ::Float64 #weight for kernel prior
    step :: Int32
    diff :: Float64 #percentage difference of x and y
end


#buffer, stop new allocations
struct PAMBuffers
    #buffers for PAM
    tv_full      :: Matrix{Float64}   #for TV function
    differencex  :: Matrix{Float64}   #For step 1
    differencexpd :: Matrix{Float64}  #DiffX buffered
    k_reversed   :: Matrix{Float64}   #convolution
    x_reversed   :: Matrix{Float64}   #for step 5
    differencek  :: Matrix{Float64}  #convolution
    diff_reversed :: Matrix{Float64}   # diffk reversed
    #buffers for !TV for no allocation
    tv_ux         :: Matrix{Float64}
    tv_uy         :: Matrix{Float64}
    tv_magnitude  :: Matrix{Float64}
    tv_normx      :: Matrix{Float64}
    tv_normy      :: Matrix{Float64}
    tv_divx       :: Matrix{Float64}
    tv_divy       :: Matrix{Float64}
end

#constructor 
function PAMBuffers(h_my::Int, w_my::Int, h::Int, w::Int)
    h_x      = h_my + (h - 1)
    w_x      = w_my + (w - 1)
    padded_h = h_my + 2*(h - 1)
    padded_w = w_my + 2*(w - 1)
    PAMBuffers(
        zeros(h_x, w_x), #tv_full
        zeros(h_my, w_my), #differencex
        zeros(padded_h, padded_w), #differencexpd
        zeros(h, w),      #k_reversed
        zeros(h_x, w_x),  #x_reversed
        zeros(h_my, w_my),#differencek
        zeros(h_my, w_my),#diff_reversed
        zeros(h_my, w_my),#tv_ux
        zeros(h_my, w_my),#tv_uy
        zeros(h_my, w_my),#rv_magnitude
        zeros(h_my, w_my),#tv_normx
        zeros(h_my, w_my),#tv_normy
        zeros(h_my, w_my),#tv_divx
        zeros(h_my, w_my),#tv_divy
    )
end



#Total Variation Function
function TV!(out::Matrix{Float64}, image::Matrix{Float64}, buf_ux::Matrix{Float64}, buf_uy::Matrix{Float64}, buf_magnitude::Matrix{Float64}, buf_normx::Matrix{Float64}, buf_normy::Matrix{Float64}, buf_divx::Matrix{Float64}, buf_divy::Matrix{Float64})
    h, w = size(image)

    #column difference by one
    buf_ux[:, 1:w-1] .= diff(image, dims=2)
    buf_ux[:, w] .= 0.0 #concatenate last row

    #row diff
    buf_uy[1:h-1, :] .= diff(image, dims=1)
    buf_uy[h, :] .= 0.0 #concatenate

    # magnitude
    buf_magnitude .= sqrt.(buf_ux.^2 .+ buf_uy.^2 .+ 1e-6) #just in case its 0
    #normalized
    buf_normx .= buf_ux ./ buf_magnitude
    buf_normy .= buf_uy ./ buf_magnitude

    #divergence: sums the derivatives of vector, need same size
    # divergence x
    buf_divx[:, 1] .= buf_normx[:, 1]
    buf_divx[:, 2:w] .= diff(buf_normx, dims=2)
    # divergence y
    buf_divy[1, :] .= buf_normy[1, :]
    buf_divy[2:h, :] .= diff(buf_normy, dims=1)

    # output wihtout allocations
    out .= buf_divx .+ buf_divy
end

# Blomgren-Chan color TV
#get gradient of pixel in each channel
#add magnitude and sum (integrate)
# divx/divy are reused for all channels

#PAM algorithm (Algorithm 1)
function run_pam(My::Matrix{Float64}, k_init::Matrix{Float64}, λ0::Float64, λmin::Float64, ϵ_x::Float64, ϵ_k::Float64, stop::Float64, max_steps::Int, x_start=nothing,  λ_now=nothing)

    h0, w0 = size(k_init)
    ph, pw = (h0-1)÷2, (w0-1)÷2
    
    if x_start === nothing #same type 3=
        #coarse
        x0 = OffsetArrays.no_offset_view(collect(padarray(My, Pad(:replicate, ph, pw))))
        x1 = Float64.(solve_ROF_PD(x0, 0.01, 20))
    else
        #fine
        x1 = x_start
    end
    #reuse kernel
    λ_now = (λ_now === nothing) ? λ0 : λ_now

    h_my, w_my = size(My)    #my
    h, w = size(k_init) #k
    h_tv, w_tv = (h-1)÷2, (w-1)÷2 #just ph, pw

    dec = Deconvolution(x1, copy(k_init), λ_now, 0.0, 0, 1e5)
    h_x, w_x = size(dec.x)
    buf = PAMBuffers(h_my, w_my, h, w)
    prev_diff = 1e3

    while (dec.diff > stop && dec.step < max_steps)
        #pre-compute single k_reversed
        copyto!(buf.k_reversed, dec.k)
        reverse!(buf.k_reversed)
        fill!(buf.tv_full, 0.0)

        #diff = k ◦ x - y
        copyto!(buf.differencex, OffsetArrays.no_offset_view(imfilter(dec.x, centered(buf.k_reversed), Inner())))
        buf.differencex .-= My
        dec.diff = norm(buf.differencex) / norm(My)
        #ratio of error distance/pixel total distance

         #4- find x
        fill!(buf.differencexpd, 0.0)
        buf.differencexpd[h:h_my+h-1, w:w_my+w-1] .= buf.differencex

        #appropriate size for TV, view is zero cost
        TV!(
            view(buf.tv_full, h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv), #out
            view(dec.x, h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv), #img
            buf.tv_ux, buf.tv_uy, buf.tv_magnitude, buf.tv_normx, buf.tv_normy, buf.tv_divx, buf.tv_divy
        )
        #just in inner x 

        #update x
        dec.x .-= ϵ_x .* OffsetArrays.no_offset_view(imfilter(buf.differencexpd, centered(dec.k), Inner()))
        dec.x .+= ϵ_x .* dec.λ .* buf.tv_full
        clamp!(dec.x, 0.0, 1.0)  #make sure to not diverge

        #5- find k
        copyto!(buf.differencek, OffsetArrays.no_offset_view(imfilter(dec.x, centered(buf.k_reversed), Inner())))
        buf.differencek .-= My

        copyto!(buf.x_reversed, dec.x)
        reverse!(buf.x_reversed)
        copyto!(buf.diff_reversed, buf.differencek)
        reverse!(buf.diff_reversed)

        dec.k .-= ϵ_k .* OffsetArrays.no_offset_view(imfilter(buf.x_reversed, centered(buf.diff_reversed), Inner())) #step 5 

        clamp!(dec.k, 1e-6, 1e6) #6
        dec.k ./= sum(dec.k) #7 normalize
        dec.λ = max(0.99*dec.λ, λmin) #8 -> will slowly decay
        dec.step += 1

        #not run forever
        if dec.step > 80 && abs(prev_diff - dec.diff) < 1e-8
            break
        end
        prev_diff = dec.diff
        if dec.step % 20 == 0
            #println(dec.step, (round(dec.diff, digits=4)), (round(dec.λ, digits=4)),  "og pam")
        end
    end

    return dec.x, dec.k, dec.λ
end