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

#Total Variation Function
function TV(image::Matrix{Float64})
    ux = diff(image, dims=2) #column difference by one
    uy = diff(image, dims=1) #row diff
    ux = hcat(ux, zeros(size(image,1), 1)) #concatenate lost row
    uy = vcat(uy, zeros(1, size(image,2)))

    magnitude = sqrt.(ux.^2 .+ uy.^2 .+ 1e-6) #just in case its 0
    normalizedx = ux ./magnitude
    normalizedy = uy ./magnitude #vectors

    div_x = hcat(normalizedx[:, 1:1], diff(normalizedx, dims=2))
    div_y = vcat(normalizedy[1:1, :], diff(normalizedy, dims=1)) #divergence: sums the derivatives of vector, need same size
    return div_x .+ div_y
end

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



    dec = Deconvolution(x1, copy(k_init), λ_now, 0.0, 0, 1e5)
    prev_diff = 1e3
    while (dec.diff > stop && dec.step < max_steps)
        #diff = k ◦ x - y
        h,w = size(dec.k)
        h_x, w_x = size(dec.x)
        tv_full = zeros(size(dec.x))

        differencex = imfilter(dec.x, centered(reverse(dec.k, dims=(1,2))), Inner())
        differencex = (OffsetArrays.no_offset_view(differencex)) .- My
        dec.diff = norm(differencex) / norm(My) #ratio of error distance/pixel total distance

        #4- find x
        differencexpd = OffsetArrays.no_offset_view(collect(padarray(differencex, Fill(0, (h-1, w-1), (h-1, w-1)))))

        #appropriate size for TV:
        h_tv = (size(dec.k, 1) - 1) ÷ 2
        w_tv = (size(dec.k, 2) - 1) ÷ 2
        tv_full[h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv] = TV(dec.x[h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv])

        dec.x = dec.x .- ϵ_x .* OffsetArrays.no_offset_view(imfilter(differencexpd, centered(dec.k), Inner())) .+ ϵ_x .* dec.λ .* tv_full #4
        dec.x = clamp.(dec.x, 0.0, 1.0) #make sure to not diverge

        #5- find k
        differencek = OffsetArrays.no_offset_view(imfilter(dec.x, centered(reverse(dec.k, dims=(1,2))), Inner())) .- My
        dec.k = dec.k .- ϵ_k .* (OffsetArrays.no_offset_view(imfilter(reverse(dec.x, dims=(1,2)), centered(reverse(differencek, dims=(1,2))), Inner()))) #5    
        dec.k = max.(dec.k, 1e-6) #6
        dec.k = dec.k ./ sum(dec.k) #7 normalize
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


function other_channelsPAM(My::Matrix{Float64}, k_init::Matrix{Float64}, λ0::Float64, λmin::Float64, ϵ_x::Float64, ϵ_k::Float64, stop::Float64, max_steps::Int, x_start=nothing,  λ_now=nothing)

    h0, w0 = size(k_init)
    ph, pw = (h0-1)÷2, (w0-1)÷2
    
    if x_start === nothing #same type 3=
        #coarse
        x0 = OffsetArrays.no_offset_view(collect(padarray(My, Pad(:replicate, ph, pw))))
        x1 = Float64.(solve_ROF_PD(x0, 0.01, 50))
    else
        #fine
        x1 = x_start
    end
    #reuse kernel
    λ_now = (λ_now === nothing) ? λ0 : λ_now

    dec = Deconvolution(x1, copy(k_init), λ_now, 0.0, 0, 1e5)
    prev_diff = 0
    while (dec.diff > stop && dec.step < max_steps)
        #diff = k ◦ x - y
        h,w = size(dec.k)
        h_x, w_x = size(dec.x)
        tv_full = zeros(size(dec.x))
        differencex = imfilter(dec.x, centered(reverse(dec.k, dims=(1,2))), Inner())
        differencex = (OffsetArrays.no_offset_view(differencex)) .- My
        dec.diff = norm(differencex) / norm(My) #ratio of error distance/pixel total distance

        #4- find x
        differencex = OffsetArrays.no_offset_view(collect(padarray(differencex, Fill(0, (h-1, w-1), (h-1, w-1)))))

        h_tv = (size(dec.k, 1) - 1) ÷ 2
        w_tv = (size(dec.k, 2) - 1) ÷ 2
        tv_full[h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv] = TV(dec.x[h_tv+1:h_x-h_tv, w_tv+1:w_x-w_tv])

        dec.x = dec.x .- ϵ_x .* OffsetArrays.no_offset_view(imfilter(differencex, centered(dec.k), Inner())) .+ ϵ_x .* dec.λ .* tv_full #4
        dec.x = clamp.(dec.x, 0.0, 1.0) #make sure to not diverge
        
        
        #5- already known k
        dec.λ = max(0.99*dec.λ, λmin) 
        dec.step += 1

        #not run forever
        if dec.step > 80 && abs(prev_diff - dec.diff) < 1e-7
            break
        end
        prev_diff = dec.diff
        if dec.step % 20 == 0
            #println(dec.step, (round(dec.diff, digits=4)), (round(dec.λ, digits=4)),  "og pam")
        end
        
    end

    return dec.x, dec.λ
end

#apply PAM to all three channels of image
function run_pam_rgb(My_rgb::Array{Float64,3}, k_init::Matrix{Float64}, λ0::Float64, λmin::Float64, ϵ_x::Float64, ϵ_k::Float64, stop::Float64, max_steps::Int, x_start = nothing, λ_now = nothing)

    _, r, c = size(My_rgb) #3
    h0, w0  = size(k_init)
    ph, pw  = (h0-1)÷2, (w0-1)÷2
    x_out   = zeros(Float64, 3, r + 2*ph, c + 2*pw)
    k_out   = copy(k_init)

    for channel in 1:3
        My_ch = My_rgb[channel, :, :] #each matrix of one colour
        xrgb_start = nothing
            if x_start !== nothing
                xrgb_start = x_start[channel, :, :]
            end        
        #First channel estimates the kernel, others use estimated channel
        #each pixel has different intensity of colour, cannot be same x -> or gray
        if channel == 1
            
            x_rgb, k_out, λ_now = run_pam(My_ch, k_init, λ0, λmin, ϵ_x, ϵ_k, stop, max_steps, xrgb_start, λ_now)
        else
            x_rgb, λ_now = other_channelsPAM(My_ch, k_out, λ0, λmin, ϵ_x, ϵ_k, stop, max_steps, xrgb_start, λ_now)
        end
        

        x_out[channel, :, :] = x_rgb
    end

    return x_out, k_out, λ_now
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

        x_out, k_now, _ = run_pam_rgb(y_level, k_now, λ0, λmin, ϵ_x, ϵ_k, stop, max_steps, x_init)
        #println("level: ", level)

    end
    #crop it out
    _, r, c = size(My_rgb)
    ph = (size(k_now, 1) - 1) ÷ 2
    pw = (size(k_now, 2) - 1) ÷ 2
    return x_out[:, 1+ph:r+ph, 1+pw:c+pw], k_now
end