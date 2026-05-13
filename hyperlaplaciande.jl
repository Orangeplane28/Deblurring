using FFTW
#inspired by https://github.com/gelpers/fastdeconv/blob/master/fastdeconv/deconv.py and paper

#Eq 5, Algorithm 3
#min_w  |w|^(2/3)  +  β/2 * (w - v)²
#maybe LUT later
#for alpha = 2/3
function compute_w(v::AbstractArray{Float64}, β::Float64)
    ϵ = 1e-6
    #quartic polynomial eq. 11:
    m  = @. 8 / (27 * β^3)
    t1 = @. -9/8 * v^2
    t2 = @. v^3 / 4
    t3 = @. -1/8 * m * v^2
    t4 = @. -t3/2 + sqrt(Complex(-m^3/27 + (m * v^2)^2 / 256))
    t5 = @. t4^(1/3)

    #t5 close to 0
    t5 = ifelse.(abs.(t5) .< ϵ, complex(ϵ), t5)


    t6 = @. 2 * (-5/18 * t1 + t5 + m / (3 * t5))
    t7 = @. sqrt(Complex(t1/3 + t6))
 
    #four roots of quartic, Algorithm 3, lines 12-15)
    r1 = @. 3v/4 + (t7 + sqrt(Complex(-(t1 + t6 + t2/t7)))) / 2
    r2 = @. 3v/4 + (t7 - sqrt(Complex(-(t1 + t6 + t2/t7)))) / 2
    r3 = @. 3v/4 + (-t7 + sqrt(Complex(-(t1 + t6 - t2/t7)))) / 2
    r4 = @. 3v/4 + (-t7 - sqrt(Complex(-(t1 + t6 - t2/t7)))) / 2
 
    sv = Float64.(sign.(v)) #sign of gradient at each pixel
    av = abs.(v)  #magnitude of gradient at each pixel
    wstar = zeros(Float64, size(v))

    #select best root: real, (1/2)|v| - |v| 
    for r in (r1, r2, r3, r4) #for all candidates
            rp    = real.(r)                 #real
            ip    = abs.(imag.(r))           #imaginary
            valid = (ip .< ϵ) .&  (rp .* sv .> 0.5 .* av) .&  (rp .* sv .< av) #valid if real, root > v/2, root < v             
            better = valid .& (rp .* sv .> wstar .* sv)   #better than current best
            wstar .= ifelse.(better, rp, wstar)      #update best at valid pixels with ifelse
        end
        return wstar
end


#last is a-d, circular padding
function circ_diff(m::AbstractMatrix{Float64}, dim::Int)
    if dim == 2
        #horizontal
        return hcat(diff(m, dims=2), m[:, 1] .- m[:, end])
    else
        return vcat(diff(m, dims=1), (m[1, :] .- m[end, :])')
    end
end

#part of eq 4
struct hyperFFT
    nomin  :: Matrix{ComplexF64}  
    denom1  :: Matrix{Float64} 
    denom2  :: Matrix{Float64}      
    i       :: Int
    j       :: Int
end

function precompute_fft(y::Matrix{Float64}, k::Matrix{Float64})
    i,j = size(y)

    #pad kernel to image size
    kp = zeros(i, j)
    ki, kj = size(k)
    kp[1:ki, 1:kj] = k #zero padding

    K = fft(kp)
    Y = fft(y)
    nomin = conj.(K) .* Y
    denom1 = abs2.(K)

    #derivative Fx=[1,-1], Fy=[1;-1] -> padded for fft
    fxp = zeros(i,j)
    fxp[1,1]=1.0; fxp[1,2]=-1.0 #Fx
    fyp = zeros(i,j)
    fyp[1,1]=1.0; fyp[2,1]=-1.0 #Fy

    Fx = fft(fxp)
    Fy = fft(fyp)

    denom2 = abs2.(Fx) .+ abs2.(Fy)
    return hyperFFT(nomin,denom1,denom2,i,j)
end

#eq 4 for x
function solve_x(fft_terms::hyperFFT, wx::Matrix{Float64}, wy::Matrix{Float64}, λ::Float64, β::Float64)
    #(F(Fx)*◦F(wx) + F(Fy)*◦F(wy))
    gradwx = -circ_diff(wx, 2); gradwy = -circ_diff(wy, 1) 
    nomin2 = fft(gradwx .+ gradwy)

    #whole equation
    nomin = fft_terms.nomin .+ (β/λ) .* nomin2
    denom = fft_terms.denom1 .+ (β/λ) .* fft_terms.denom2

    x = real.(ifft(nomin ./ denom))
    return x
end


#Algorithm 1
#Must minimize both: xk - y, and Fx - w 
#y,k
#λ = regularization weight between data and hl prior, =10(?)
#α = 2/3-> hyper laplacian for strong edges 
#β​0 = initial constraint on w = Fx
#β​rate = slower for stable convergence
function hyperlaplacian_deconv(y::Matrix{Float64}, k::Matrix{Float64}, λ::Float64; α::Float64 = 2/3, β0::Float64 = 1.0, β_rate::Float64 = 2*sqrt(2), β_max::Float64 = 256.0)
   #1: initial vars
    k = k./sum(k)
    β = β0
    iter = 0
    x = copy(y)
    T = 1 #could have more iterations if necessary

    #2: precompute terms
    ffterms = precompute_fft(y,k)
    iter = 0
    #3: while beta loop
    while (β < β_max)
        
        #4: iter 0
        #5: iter to T
        for _ in 1:T
            gradx = circ_diff(x, 2) #∇x
            grady = circ_diff(x, 1) #∇y
            #6: solve w, alg 3
            wx = compute_w(gradx, β)
            wy = compute_w(grady, β)

            #7: solve x, eq 4
            x = solve_x(ffterms, wx, wy, λ, β)   
        end
            #8: increment
            β *= β_rate
            iter += 1
    end

    kh, kw = size(k)
    x = circshift(x, (kh÷2, kw÷2))
    clamp!(x, 0.0, 1.0)

    println("Hyper-Laplacian: $(iter*T) total iters, α=$(α)")
    return x
end
                            

#apply to all channels
function hyperlaplacian_deconv_rgb(My_rgb::Array{Float64,3}, k::Matrix{Float64}, λ::Float64; α::Float64 = 2/3, β0::Float64 = 1.0, β_rate::Float64 = 2*sqrt(2), β_max::Float64 = 256.0)
    _, H, W = size(My_rgb)
    out = zeros(3, H, W)
    for ch in 1:3
        out[ch, :, :] = hyperlaplacian_deconv(My_rgb[ch, :, :], k, λ,α=α, β0=β0, β_rate=β_rate, β_max=β_max)
    end
    return out
end