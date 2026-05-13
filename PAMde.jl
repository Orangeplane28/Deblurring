#made inspired by PAM paper
using ImageFiltering
using ImageFiltering.Models
using LinearAlgebra
using FileIO, Images
using OffsetArrays
using Images
using FFTW

FFTW.set_num_threads(8) 

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
    #tv_magnitude  :: Matrix{Float64}
    #tv_normx      :: Matrix{Float64}
    #tv_normy      :: Matrix{Float64}
    #tv_divx       :: Matrix{Float64}
    #tv_divy       :: Matrix{Float64}
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
        #zeros(h_my, w_my),#rv_magnitude
        #zeros(h_my, w_my),#tv_normx
        #zeros(h_my, w_my),#tv_normy
        #zeros(h_my, w_my),#tv_divx
        #zeros(h_my, w_my),#tv_divy
    )
end

struct ChannelBuffers
    tv_full :: Matrix{Float64}
    differencex :: Matrix{Float64}
    differencexpd :: Matrix{Float64}
    tv_ux :: Matrix{Float64}
    tv_uy :: Matrix{Float64}
end

function ChannelBuffers(h_my::Int, w_my::Int, h::Int, w::Int)
    h_x      = h_my + (h - 1)
    w_x      = w_my + (w - 1)
    padded_h = h_my + 2*(h - 1)
    padded_w = w_my + 2*(w - 1)
    ChannelBuffers(
        zeros(h_x, w_x),
        zeros(h_my, w_my),
        zeros(padded_h, padded_w),
        zeros(h_my, w_my),
        zeros(h_my, w_my),
    )
end

# shared buffers for vectorial TV
struct VectorialTVBuffers
    magnitude :: Matrix{Float64}
    divx :: Matrix{Float64}
    divy :: Matrix{Float64}
    tv_r :: Matrix{Float64}
    tv_g :: Matrix{Float64}
    tv_b :: Matrix{Float64}
end

function VectorialTVBuffers(h_my::Int, w_my::Int)
    VectorialTVBuffers(
        zeros(h_my, w_my),
        zeros(h_my, w_my),
        zeros(h_my, w_my),
        zeros(h_my, w_my),
        zeros(h_my, w_my),
        zeros(h_my, w_my),
    )
end

# Blomgren-Chan color TV
#get gradient of pixel in each channel
#add magnitude and sum (integrate)
# divx/divy are reused for all channels
function vectorial_TV!(xr::AbstractMatrix{Float64},
                        xg::AbstractMatrix{Float64},
                        xb::AbstractMatrix{Float64},
                        ux_r::Matrix{Float64}, uy_r::Matrix{Float64},
                        ux_g::Matrix{Float64}, uy_g::Matrix{Float64},
                        ux_b::Matrix{Float64}, uy_b::Matrix{Float64},
                        vbuf::VectorialTVBuffers)
    h, w = size(xr)

    # step 1: gradients for all channels
    ux_r[:, 1:w-1] .= diff(xr, dims=2);  ux_r[:, w] .= 0.0
    uy_r[1:h-1, :] .= diff(xr, dims=1);  uy_r[h, :] .= 0.0

    ux_g[:, 1:w-1] .= diff(xg, dims=2);  ux_g[:, w] .= 0.0
    uy_g[1:h-1, :] .= diff(xg, dims=1);  uy_g[h, :] .= 0.0

    ux_b[:, 1:w-1] .= diff(xb, dims=2);  ux_b[:, w] .= 0.0
    uy_b[1:h-1, :] .= diff(xb, dims=1);  uy_b[h, :] .= 0.0

    # step 2: shared magnitude
    vbuf.magnitude .= sqrt.(
        ux_r.^2 .+ uy_r.^2 .+
        ux_g.^2 .+ uy_g.^2 .+
        ux_b.^2 .+ uy_b.^2 .+ 1e-6
    )

    #channel divergence, normalize and reuse divx/divy
    ux_r ./= vbuf.magnitude #norm
    uy_r ./= vbuf.magnitude
    vbuf.divx[:, 1]   .= ux_r[:, 1]
    vbuf.divx[:, 2:w] .= diff(ux_r, dims=2)
    vbuf.divy[1, :]   .= uy_r[1, :]
    vbuf.divy[2:h, :] .= diff(uy_r, dims=1)
    vbuf.tv_r         .= vbuf.divx .+ vbuf.divy

    ux_g ./= vbuf.magnitude #norm
    uy_g ./= vbuf.magnitude
    vbuf.divx[:, 1]   .= ux_g[:, 1]
    vbuf.divx[:, 2:w] .= diff(ux_g, dims=2)
    vbuf.divy[1, :]   .= uy_g[1, :]
    vbuf.divy[2:h, :] .= diff(uy_g, dims=1)
    vbuf.tv_g         .= vbuf.divx .+ vbuf.divy

    ux_b ./= vbuf.magnitude #norm
    uy_b ./= vbuf.magnitude
    vbuf.divx[:, 1]   .= ux_b[:, 1]
    vbuf.divx[:, 2:w] .= diff(ux_b, dims=2)
    vbuf.divy[1, :]   .= uy_b[1, :]
    vbuf.divy[2:h, :] .= diff(uy_b, dims=1)
    vbuf.tv_b         .= vbuf.divx .+ vbuf.divy
end

#PAM algorithm (Algorithm 1) with vectorial TV instead at every step
function run_pam_rgb(My_rgb::Array{Float64,3}, k_init::Matrix{Float64}, λ0::Float64, λmin::Float64, ϵ_x::Float64, ϵ_k::Float64, stop::Float64, max_steps::Int, x_start=nothing, λ_now=nothing)

    _, r, c = size(My_rgb)
    h, w = size(k_init)
    h_tv, w_tv = (h-1)÷2, (w-1)÷2
    h_my, w_my = r, c

    λ_now = (λ_now === nothing) ? λ0 : λ_now
    My_r = My_rgb[1, :, :]
    My_g = My_rgb[2, :, :]
    My_b = My_rgb[3, :, :]
    norm_My = norm(My_r) + norm(My_g) + norm(My_b)

    #initialize x
    function init_x(My_ch, rof_iters, xstart)
        if xstart === nothing
            x0 = OffsetArrays.no_offset_view(collect(padarray(My_ch, Pad(:replicate, h_tv, w_tv))))
            return Float64.(solve_ROF_PD(x0, 0.01, rof_iters))
        else
            return xstart
        end
    end

    xstart_r = x_start !== nothing ? x_start[1, :, :] : nothing
    xstart_g = x_start !== nothing ? x_start[2, :, :] : nothing
    xstart_b = x_start !== nothing ? x_start[3, :, :] : nothing

    #R does PAM and G and B only need x
    dec   = Deconvolution(init_x(My_r, 20, xstart_r), copy(k_init), λ_now, 0.0, 0, 1e5)
    x_g   = init_x(My_g, 50, xstart_g)
    x_b   = init_x(My_b, 50, xstart_b)

    h_x, w_x = size(dec.x)

    #buffers
    buf_r  = PAMBuffers(h_my, w_my, h, w)
    buf_g  = ChannelBuffers(h_my, w_my, h, w)
    buf_b  = ChannelBuffers(h_my, w_my, h, w)
    vbuf   = VectorialTVBuffers(h_my, w_my)

    prev_diff = 1e3
    step      = 0
    diff_val  = 1e5

    while (diff_val > stop && step < max_steps)

        #reverse k
        copyto!(buf_r.k_reversed, dec.k)
        reverse!(buf_r.k_reversed, dims = 1)
        reverse!(buf_r.k_reversed, dims=2)

        #residual
        copyto!(buf_r.differencex, OffsetArrays.no_offset_view(imfilter(dec.x, centered(buf_r.k_reversed), Inner())))
        buf_r.differencex .-= My_r

        copyto!(buf_g.differencex, OffsetArrays.no_offset_view(imfilter(x_g, centered(buf_r.k_reversed), Inner())))
        buf_g.differencex .-= My_g

        copyto!(buf_b.differencex, OffsetArrays.no_offset_view(imfilter(x_b, centered(buf_r.k_reversed), Inner())))
        buf_b.differencex .-= My_b

        diff_val = (norm(buf_r.differencex) + norm(buf_g.differencex) + norm(buf_b.differencex)) / norm_My

        #padded
        fill!(buf_r.differencexpd, 0.0)
        buf_r.differencexpd[h:h_my+h-1, w:w_my+w-1] .= buf_r.differencex

        fill!(buf_g.differencexpd, 0.0)
        buf_g.differencexpd[h:h_my+h-1, w:w_my+w-1] .= buf_g.differencex

        fill!(buf_b.differencexpd, 0.0)
        buf_b.differencexpd[h:h_my+h-1, w:w_my+w-1] .= buf_b.differencex

        #vectorial TV for al lchannels
        vectorial_TV!(
        view(dec.x, h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv),
        view(x_g,   h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv),
        view(x_b,   h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv),
        buf_r.tv_ux, buf_r.tv_uy, buf_g.tv_ux, buf_g.tv_uy, buf_b.tv_ux, buf_b.tv_uy,
        vbuf
        )

        #tv_full buffers
        fill!(buf_r.tv_full, 0.0)
        buf_r.tv_full[h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv] .= vbuf.tv_r

        fill!(buf_g.tv_full, 0.0)
        buf_g.tv_full[h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv] .= vbuf.tv_g

        fill!(buf_b.tv_full, 0.0)
        buf_b.tv_full[h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv] .= vbuf.tv_b

        #x update for all channels
        dec.x .-= ϵ_x .* OffsetArrays.no_offset_view(imfilter(buf_r.differencexpd, centered(dec.k), Inner()))
        dec.x .+= ϵ_x .* dec.λ .* buf_r.tv_full
        clamp!(dec.x, 0.0, 1.0)

        x_g .-= ϵ_x .* OffsetArrays.no_offset_view(imfilter(buf_g.differencexpd, centered(dec.k), Inner()))
        x_g .+= ϵ_x .* dec.λ .* buf_g.tv_full
        clamp!(x_g, 0.0, 1.0)

        x_b .-= ϵ_x .* OffsetArrays.no_offset_view(imfilter(buf_b.differencexpd, centered(dec.k), Inner()))
        x_b .+= ϵ_x .* dec.λ .* buf_b.tv_full
        clamp!(x_b, 0.0, 1.0)

        #k update from R only
        copyto!(buf_r.differencek, OffsetArrays.no_offset_view(imfilter(dec.x, centered(buf_r.k_reversed), Inner())))
        buf_r.differencek .-= My_r

        copyto!(buf_r.x_reversed, dec.x)
        reverse!(buf_r.x_reversed, dims=1)
        reverse!(buf_r.x_reversed, dims=2)
        copyto!(buf_r.diff_reversed, buf_r.differencek)
        reverse!(buf_r.diff_reversed, dims=1)
        reverse!(buf_r.diff_reversed, dims=2)

        dec.k .-= ϵ_k .* OffsetArrays.no_offset_view(imfilter(buf_r.x_reversed, centered(buf_r.diff_reversed), Inner()))

        #step 6,7,8
        clamp!(dec.k, 1e-6, Inf)
        dec.k ./= sum(dec.k)
        dec.λ = max(0.99 * dec.λ, λmin)

        step += 1
        if step > 80 && abs(prev_diff - diff_val) / (diff_val + 1e-12) < 1e-8
            break
        end
        prev_diff = diff_val
    end

    x_out = zeros(Float64, 3, r + 2*h_tv, c + 2*w_tv)
    x_out[1, :, :] = dec.x
    x_out[2, :, :] = x_g
    x_out[3, :, :] = x_b

    return x_out, dec.k, dec.λ
end

#now pyramid scheme optimization: coarse -> fine
function run_pyramid_rgb(My_rgb::Array{Float64,3}, k_init::Matrix{Float64}, λ0::Float64, λmin::Float64, ϵ_x::Float64, ϵ_k::Float64, stop::Float64, max_coarse::Int, max_fine::Int)
    n = size(k_init, 1)

    #build pyrami
    pyramid = [My_rgb] #array that will be pushed into
    Mcurr = My_rgb #current

    while true
        _, h, w  = size(Mcurr)
        next_h, next_w = h ÷ 2, w ÷ 2 
        nr_ksize = n * (next_h / size(My_rgb, 2)) #coarser by one level, halve i5 and same for kernel
        if nr_ksize < 3 || next_h < 16 || next_w < 16
            break
        end
        next_img = zeros(Float64, 3, next_h, next_w)
        for ch in 1:3
            next_img[ch, :, :] = imresize(Mcurr[ch, :, :], (next_h, next_w))  #resize each channel seperately
        end
        push!(pyramid, next_img) #add it
        Mcurr = next_img
    end

    pyramid = reverse(pyramid) #coarse to fine
    n_levels = length(pyramid)

    # uniform init, no spike
    k_now = fill(1.0/9, 3, 3)
    x_out = nothing
    λ_now = nothing

    #for each level, 
    for level in 1:n_levels
        y_level = pyramid[level] #blurreed from each resolution
        _, img_h, img_w = size(y_level)

        scale = img_h / size(My_rgb, 2) #resolution ratio
        kn_level = max(3, round(Int, n * scale)) #minimum of 3
        kn_level = isodd(kn_level) ? kn_level : kn_level + 1 #must be odd

        ϵ_k_level = ϵ_k / (scale^2) #inversely proprotional

        if size(k_now) != (kn_level, kn_level)
            k_now = imresize(k_now, (kn_level, kn_level))
            k_now = max.(k_now, 0.00000001) #keep pixels positive
            k_now = k_now ./ sum(k_now)  #normalize
        end

        max_steps = (level == n_levels) ? max_fine : max_coarse #less computationally expensive

        # upsample x_out
        x_init = nothing
        if x_out !== nothing #prepare correct values after initialization
            ph = (kn_level - 1) ÷ 2
            pw = (kn_level - 1) ÷ 2
            x_level = zeros(Float64, 3, img_h + 2*ph, img_w + 2*pw)
            for ch in 1:3
                x_level[ch, :, :] = imresize(x_out[ch, :, :], (img_h + 2*ph, img_w + 2*pw)) #upsample
            end
            x_init = x_level
        end

        x_out, k_now, _ = run_pam_rgb(y_level, k_now, λ0, λmin, ϵ_x, ϵ_k_level, stop, max_steps, x_init)
        println("Level $level: img=$(img_h)×$(img_w)  k=$(kn_level)×$(kn_level)  steps=$max_steps  k_max=$(round(maximum(k_now), digits=6))")

    end
    #crop it out
    _, r, c = size(My_rgb)
    ph = (size(k_now, 1) - 1) ÷ 2
    pw = (size(k_now, 2) - 1) ÷ 2
    return x_out[:, 1+ph:r+ph, 1+pw:c+pw], k_now
end