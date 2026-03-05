using ImageFiltering
using ImageFiltering.Models
using LinearAlgebra
using FileIO, Images
using OffsetArrays
using Images

#norm() = sqrt(sum(().^2)) = euclidean distance
#imfiler -> correlation = flip kernel for convolution

mutable struct Deconvolution 
    x :: Matrix{Float64}
    k :: Matrix{Float64}
    λ::Float64 #weight for TV
    γ::Float64 #weight for kernel prior
    step :: Int32
    diff :: Float64 #percentage difference of x and y
end

function TV(image::Matrix{Float64})
    ux = diff(image, dims=2) #column difference by one
    uy = diff(image, dims=1) #row diff
    ux = hcat(ux, zeros(size(image,1), 1)) #concatenate lost row
    uy = vcat(uy, zeros(1, size(image,2)))

    magnitude = sqrt.(ux.^2 .+ uy.^2 .+ 0.0000000001) #just in case its 0
    normalizedx = ux ./magnitude
    normalizedy = uy ./magnitude #vectors

    div_x = hcat(normalizedx[:, 1:1], diff(normalizedx, dims=2))
    div_y = vcat(normalizedy[1:1, :], diff(normalizedy, dims=1)) #divergence: sums the derivatives of vector, need same size
    return div_x .+ div_y
end



gety = load("C:/Users/User/Downloads/Deblurring/data/train/blur/0.png")
My = Float64.(Gray.(gety)) #convert to grayscale

#algorithm 1:
λmin = 0.0006
λ0 = 0.1 #start high value

n=7
k0 = fill(1/(n*n), n, n)
k0[(n+1)÷2,(n+1)÷2] = 1.0
k0 = k0 ./ sum(k0)

h0,w0 = size(k0)
ph, pw = (h0-1)÷2, (w0-1)÷2
x0 = OffsetArrays.no_offset_view(collect(padarray(My, Pad(:replicate, ph, pw))))
x1 = solve_ROF_PD(x0, 0.01, 50)
dec = Deconvolution(x1, k0, λ0, 0.0, 0, 1e5)
ϵ_x = 1e-2 
ϵ_k = 1e-4

#x1 = solve_ROF_PD(My, 0.01, 50)  1: TV denoising

while (dec.diff > 0.001 && dec.step < 100 ) #until not converged or maxmum steps
    #diff = k ◦ x - y
    h,w = size(dec.k)
    differencex = imfilter(dec.x, centered(reverse(dec.k, dims=(1,2))), Inner())
    differencex = (OffsetArrays.no_offset_view(differencex)) .- My
    dec.diff = norm(differencex) / norm(My) #ratio of error distance/pixel total distance
    
    #4- find x
    differencex = OffsetArrays.no_offset_view(collect(padarray(differencex, Fill(0, (h-1, w-1), (h-1, w-1)))))
    dec.x = dec.x .- ϵ_x .* OffsetArrays.no_offset_view(imfilter(differencex, centered(dec.k), Inner())) .+ ϵ_x .* dec.λ .* TV(dec.x) #4

    #5- find k
    differencek = OffsetArrays.no_offset_view(imfilter(dec.x, centered(reverse(dec.k, dims=(1,2))), Inner())) .- My
    dec.k = dec.k .- ϵ_k .* (OffsetArrays.no_offset_view(imfilter(reverse(dec.x, dims=(1,2)), centered(reverse(differencek, dims=(1,2))), Inner()))) #5    
    dec.k = max.(dec.k, 0.00000001) #6
    dec.k = dec.k ./ sum(dec.k) #7 normalize
    dec.λ = max(0.99*dec.λ, λmin) #8 -> will slowly decay
    dec.step += 1
    #done
end


println("Converged after $(dec.step) steps, final diff: $(dec.diff)")
save("restored.png", clamp.(dec.x, 0, 1))
save("kernel.png", dec.k ./ maximum(dec.k))