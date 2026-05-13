# =============================================================================
#  LocalPatchTV.jl
#
#  Non-blind local patch deblurring via ADMM.
#
#      min_{x ≥ 0}  ½‖H(x) − y‖²  +  λ · R(x)
#
#  where:
#     H(x)  = non-circular local blur:  pad with zeros, convolve with k, crop
#     Hᵀ(r) = same-sized zero-boundary correlation with k  (exact adjoint)
#     R(x)  = isotropic TV  (default) or  isotropic Huber-TV (huber_δ > 0)
#
#  Variable splitting:
#     z = Dx           (for the TV prior)
#     w = x            (for the positivity / box constraint)
#
#  x-subproblem is (HᵀH + ρ₁DᵀD + ρ₂I) x = b, solved with a warm-started
#  conjugate-gradient inner loop.  z-subproblem is isotropic shrinkage
#  (exact prox, no smoothing).  w-subproblem is a clamp projection.
#
#  No FFT, no wraparound.  Fully spatial, drop-in per patch.
# =============================================================================

module LocalPatchTV

using ImageFiltering
using OffsetArrays
using LinearAlgebra
using Printf

export solve_local_patch_tv, validate_kernel, LocalTVInfo

# -----------------------------------------------------------------------------
#  Info struct returned alongside the reconstruction.
# -----------------------------------------------------------------------------
struct LocalTVInfo
    iters            :: Int
    converged        :: Bool
    reason           :: Symbol            # :primal_dual | :objective | :max_iters
    obj_history      :: Vector{Float64}
    primal_history   :: Vector{Float64}
    dual_history     :: Vector{Float64}
    rel_change_history :: Vector{Float64}
end

# -----------------------------------------------------------------------------
#  Kernel validation and normalization.
# -----------------------------------------------------------------------------
"""
    validate_kernel(k; nonneg=true, normalize=true, atol=1e-12) -> Matrix{Float64}

Checks that `k` is a 2-D, finite, odd-sized kernel.  Optionally verifies
nonnegativity and normalizes to sum-to-one.  Returns a fresh `Float64` copy.
"""
function validate_kernel(k::AbstractMatrix{<:Real};
                         nonneg::Bool    = true,
                         normalize::Bool = true,
                         atol::Float64   = 1e-12)
    @assert ndims(k) == 2 "kernel must be 2D"
    all(isfinite, k) || throw(ArgumentError("kernel contains NaN or Inf"))
    h_k, w_k = size(k)
    (isodd(h_k) && isodd(w_k)) ||
        throw(ArgumentError("kernel dims must be odd (got $h_k × $w_k)"))
    if nonneg && minimum(k) < -atol
        throw(ArgumentError("kernel has negative entries (min = $(minimum(k)))"))
    end
    k_out = Matrix{Float64}(k)
    if normalize
        s = sum(k_out)
        s > atol || throw(ArgumentError("kernel sums to ≈ 0 (got $s); cannot normalize"))
        k_out ./= s
    end
    return k_out
end

# -----------------------------------------------------------------------------
#  Non-circular blur operator and its exact adjoint.
#
#  imfilter is correlation.  Passing k_flip gives convolution with k,
#  i.e. the "pad-zero → conv → crop" forward blur H.  Same-sized output
#  with Fill(0) is equivalent to zero-padding, full conv, and cropping.
#  H^T under the same zero boundary is correlation with k — another
#  same-sized Fill(0) imfilter with the un-flipped kernel.
# -----------------------------------------------------------------------------
@inline apply_H(x::AbstractMatrix, k_flip::AbstractMatrix) =
    OffsetArrays.no_offset_view(imfilter(x, centered(k_flip), Fill(0.0)))

@inline apply_Ht(r::AbstractMatrix, k::AbstractMatrix) =
    OffsetArrays.no_offset_view(imfilter(r, centered(k), Fill(0.0)))

# -----------------------------------------------------------------------------
#  Discrete forward-difference gradient and its exact adjoint.
#  "Zero beyond boundary":  Dx[:,end] = 0,  Dy[end,:] = 0.
#  Under this convention, Dᵀ p is:
#      (Dᵀp)_{i,1}   = -p_x[i,1]
#      (Dᵀp)_{i,j}   =  p_x[i,j-1] − p_x[i,j]    j ≥ 2
#      + analogous for the y-component
#  (The uniform j ≥ 2 form also holds at j = w because p_x[i,w] = 0 by
#  construction of grad!.)
# -----------------------------------------------------------------------------
function grad!(Dx::AbstractMatrix{Float64},
               Dy::AbstractMatrix{Float64},
               x::AbstractMatrix{Float64})
    h, w = size(x)
    @inbounds for j in 1:w-1, i in 1:h
        Dx[i, j] = x[i, j+1] - x[i, j]
    end
    @inbounds for i in 1:h
        Dx[i, w] = 0.0
    end
    @inbounds for j in 1:w, i in 1:h-1
        Dy[i, j] = x[i+1, j] - x[i, j]
    end
    @inbounds for j in 1:w
        Dy[h, j] = 0.0
    end
    return nothing
end

function Dt!(out::AbstractMatrix{Float64},
             px::AbstractMatrix{Float64},
             py::AbstractMatrix{Float64})
    h, w = size(out)
    @inbounds for j in 1:w, i in 1:h
        vx = (j == 1) ? -px[i, 1] : (px[i, j-1] - px[i, j])
        vy = (i == 1) ? -py[1, j] : (py[i-1, j] - py[i, j])
        out[i, j] = vx + vy
    end
    return nothing
end

# -----------------------------------------------------------------------------
#  Reusable buffers.  All arrays are y-patch sized, allocated once per
#  solver call.  A pipeline driver can also instantiate one ADMMBuffers
#  per patch-size class and pass it in (future extension).
# -----------------------------------------------------------------------------
struct ADMMBuffers
    # primals / slacks
    zx      :: Matrix{Float64}
    zy      :: Matrix{Float64}
    zx_prev :: Matrix{Float64}
    zy_prev :: Matrix{Float64}
    w       :: Matrix{Float64}
    w_prev  :: Matrix{Float64}
    # scaled duals
    u1x     :: Matrix{Float64}
    u1y     :: Matrix{Float64}
    u2      :: Matrix{Float64}
    # working buffers
    Hty     :: Matrix{Float64}          # H^T y (precomputed once)
    rhs     :: Matrix{Float64}          # RHS of x-subproblem
    Dx      :: Matrix{Float64}
    Dy      :: Matrix{Float64}
    Dt_buf  :: Matrix{Float64}          # scratch for D^T of something
    x_prev  :: Matrix{Float64}          # previous x (for rel-change stop)
    # CG
    cg_r    :: Matrix{Float64}
    cg_p    :: Matrix{Float64}
    cg_Ap   :: Matrix{Float64}
    cg_Ax   :: Matrix{Float64}
    cg_tmp1 :: Matrix{Float64}
    cg_tmp2 :: Matrix{Float64}
    cg_tmp3 :: Matrix{Float64}
    cg_tmp4 :: Matrix{Float64}
end

function ADMMBuffers(H::Int, W::Int)
    mk() = zeros(H, W)
    ADMMBuffers(
        mk(), mk(), mk(), mk(),
        mk(), mk(),
        mk(), mk(), mk(),
        mk(), mk(),
        mk(), mk(), mk(),
        mk(),
        mk(), mk(), mk(), mk(),
        mk(), mk(), mk(), mk(),
    )
end

# -----------------------------------------------------------------------------
#  Operator A v = H^T H v  +  ρ₁ D^T D v  +  ρ₂ v.
# -----------------------------------------------------------------------------
function apply_A!(out::AbstractMatrix{Float64},
                  v::AbstractMatrix{Float64},
                  k::AbstractMatrix{Float64},
                  k_flip::AbstractMatrix{Float64},
                  ρ1::Float64, ρ2::Float64,
                  buf::ADMMBuffers)
    # H^T H v
    Hv   = apply_H(v, k_flip)
    HtHv = apply_Ht(Hv, k)
    # D^T D v
    grad!(buf.Dx, buf.Dy, v)
    Dt!(buf.cg_tmp1, buf.Dx, buf.Dy)
    @. out = HtHv + ρ1 * buf.cg_tmp1 + ρ2 * v
    return nothing
end

# -----------------------------------------------------------------------------
#  Warm-started CG for the x-subproblem.  Inexact solves are fine for ADMM.
# -----------------------------------------------------------------------------
function cg_solve!(x::AbstractMatrix{Float64},
                   b::AbstractMatrix{Float64},
                   k::AbstractMatrix{Float64},
                   k_flip::AbstractMatrix{Float64},
                   ρ1::Float64, ρ2::Float64,
                   buf::ADMMBuffers;
                   maxit::Int = 25, tol::Float64 = 1e-6)
    apply_A!(buf.cg_Ax, x, k, k_flip, ρ1, ρ2, buf)
    @. buf.cg_r = b - buf.cg_Ax
    @. buf.cg_p = buf.cg_r
    rsold = dot(buf.cg_r, buf.cg_r)
    r0    = sqrt(rsold) + 1e-30
    iters = 0
    for it in 1:maxit
        apply_A!(buf.cg_Ap, buf.cg_p, k, k_flip, ρ1, ρ2, buf)
        pAp = dot(buf.cg_p, buf.cg_Ap)
        α   = rsold / (pAp + 1e-30)
        @. x         += α * buf.cg_p
        @. buf.cg_r -= α * buf.cg_Ap
        rsnew = dot(buf.cg_r, buf.cg_r)
        iters = it
        sqrt(rsnew) / r0 < tol && break
        β = rsnew / rsold
        @. buf.cg_p = buf.cg_r + β * buf.cg_p
        rsold = rsnew
    end
    return iters
end

# -----------------------------------------------------------------------------
#  Isotropic TV / Huber-TV prox (per-pixel on the 2-vector (vx, vy)).
#
#     TV:          z = max(0, 1 − t/‖v‖) · v                     (huber_δ ≤ 0)
#     Huber-TV:    z = δ/(δ+t) · v                      if ‖v‖ ≤ δ+t
#                  z = (1 − t/‖v‖) · v                  otherwise
#
#  t here is the prox step, i.e. λ/ρ₁.  δ is the Huber knee in ‖∇x‖.
# -----------------------------------------------------------------------------
function tv_prox!(zx::AbstractMatrix{Float64}, zy::AbstractMatrix{Float64},
                  vx::AbstractMatrix{Float64}, vy::AbstractMatrix{Float64},
                  t::Float64, huber_δ::Float64)
    if huber_δ <= 0.0
        @inbounds for i in eachindex(zx)
            m = sqrt(vx[i]^2 + vy[i]^2)
            s = m > t ? (1.0 - t / m) : 0.0
            zx[i] = s * vx[i]
            zy[i] = s * vy[i]
        end
    else
        cutoff = huber_δ + t
        r_quad = huber_δ / cutoff
        @inbounds for i in eachindex(zx)
            m = sqrt(vx[i]^2 + vy[i]^2)
            if m <= cutoff
                zx[i] = r_quad * vx[i]
                zy[i] = r_quad * vy[i]
            else
                s = 1.0 - t / m
                zx[i] = s * vx[i]
                zy[i] = s * vy[i]
            end
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
#  Objective:   ½‖Hx − y‖² + λ · R(x).
# -----------------------------------------------------------------------------
function objective(x::AbstractMatrix{Float64},
                   y::AbstractMatrix{Float64},
                   k_flip::AbstractMatrix{Float64},
                   λ::Float64, huber_δ::Float64,
                   buf::ADMMBuffers)
    Hx  = apply_H(x, k_flip)
    dat = 0.5 * sum(abs2, Hx .- y)
    grad!(buf.Dx, buf.Dy, x)
    reg = 0.0
    if huber_δ <= 0.0
        @inbounds for i in eachindex(buf.Dx)
            reg += sqrt(buf.Dx[i]^2 + buf.Dy[i]^2)
        end
    else
        @inbounds for i in eachindex(buf.Dx)
            m = sqrt(buf.Dx[i]^2 + buf.Dy[i]^2)
            reg += m <= huber_δ ? (m^2) / (2 * huber_δ) : (m - huber_δ / 2)
        end
    end
    return dat + λ * reg
end

# -----------------------------------------------------------------------------
#  Main solver.
# -----------------------------------------------------------------------------
"""
    solve_local_patch_tv(y_patch, k_local, λ; kwargs...) -> (x, info)

Non-blind, local, non-circular TV deblurring of a single grayscale patch via
ADMM with an inner CG solve.

# Arguments
- `y_patch :: AbstractMatrix{<:Real}`  observed blurred patch, H × W.
- `k_local :: AbstractMatrix{<:Real}`  fixed local blur kernel (odd × odd).
- `λ :: Real`                          TV weight.

# Keyword arguments
- `x0::Union{Nothing,AbstractMatrix}` = nothing     warm start (H × W); default `y_patch`
- `ρ1::Real` = 1.0                                  penalty on  Dx − z
- `ρ2::Real` = 1.0                                  penalty on  x  − w
- `max_iters::Int` = 200                            outer ADMM iterations
- `abs_tol::Real` = 1e-5                            absolute residual tol  (Boyd)
- `rel_tol::Real` = 1e-4                            relative residual tol (Boyd)
- `obj_tol::Real` = 1e-6                            rel-objective-change stop
- `xchange_tol::Real` = 0.0                         rel ‖Δx‖ / ‖x‖ stop (0 disables)
- `cg_maxit::Int` = 25                              CG iterations per x-update
- `cg_tol::Real` = 1e-6                             CG relative tolerance
- `clamp_hi::Real` = 1.0                            upper clamp for [0, clamp_hi]
- `huber_δ::Real` = 0.0                             > 0 ⇒ isotropic Huber-TV prior
- `validate::Bool` = true                           run `validate_kernel`
- `normalize_k::Bool` = true                        normalize kernel to sum-1
- `verbose::Bool` = false                           per-iteration printout

Returns `(x::Matrix{Float64}, info::LocalTVInfo)`.
"""
function solve_local_patch_tv(y_patch::AbstractMatrix{<:Real},
                              k_local::AbstractMatrix{<:Real},
                              λ::Real;
                              x0::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                              ρ1::Real       = 1.0,
                              ρ2::Real       = 1.0,
                              max_iters::Int = 200,
                              abs_tol::Real  = 1e-5,
                              rel_tol::Real  = 1e-4,
                              obj_tol::Real  = 1e-6,
                              xchange_tol::Real = 0.0,
                              cg_maxit::Int  = 25,
                              cg_tol::Real   = 1e-6,
                              clamp_hi::Real = 1.0,
                              huber_δ::Real  = 0.0,
                              validate::Bool = true,
                              normalize_k::Bool = true,
                              verbose::Bool  = false)

    @assert ρ1 > 0 && ρ2 > 0 "ρ1, ρ2 must be positive"
    @assert λ   ≥ 0          "λ must be nonnegative"
    @assert max_iters ≥ 1

    y        = Matrix{Float64}(y_patch)
    k        = validate ? validate_kernel(k_local; normalize = normalize_k) :
                          Matrix{Float64}(k_local)
    λf       = Float64(λ)
    ρ1f      = Float64(ρ1)
    ρ2f      = Float64(ρ2)
    clamp_hi_f = Float64(clamp_hi)
    huber_δ_f  = Float64(huber_δ)

    H_y, W_y = size(y)
    x = x0 === nothing ? copy(y) : Matrix{Float64}(x0)
    size(x) == size(y) || throw(DimensionMismatch("x0 size must match y_patch"))
    clamp!(x, 0.0, clamp_hi_f)

    k_flip = reverse(reverse(k, dims = 1), dims = 2)
    buf    = ADMMBuffers(H_y, W_y)

    # Precompute H^T y (constant across outer iterations).
    copyto!(buf.Hty, apply_Ht(y, k))

    # Initialize splits consistently:  z = Dx, w = x, duals = 0.
    grad!(buf.Dx, buf.Dy, x)
    buf.zx .= buf.Dx
    buf.zy .= buf.Dy
    buf.w  .= x
    fill!(buf.u1x, 0.0); fill!(buf.u1y, 0.0); fill!(buf.u2, 0.0)

    obj_hist = Float64[]
    pri_hist = Float64[]
    dua_hist = Float64[]
    relx_hist = Float64[]

    N       = length(x)
    sqrt_p  = sqrt(3 * N)    # constraints have 2N (Dx−z) + N (x−w) = 3N
    sqrt_n  = sqrt(N)        # primal dim

    prev_obj  = Inf
    converged = false
    reason    = :max_iters
    iters_done = 0

    for it in 1:max_iters
        iters_done = it
        buf.x_prev .= x

        # --- x-subproblem: (H^T H + ρ₁ D^T D + ρ₂ I) x = rhs ----------------
        # rhs = H^T y + ρ₁ D^T (z − u1) + ρ₂ (w − u2)
        @. buf.cg_tmp3 = buf.zx - buf.u1x
        @. buf.cg_tmp4 = buf.zy - buf.u1y
        Dt!(buf.Dt_buf, buf.cg_tmp3, buf.cg_tmp4)
        @. buf.rhs = buf.Hty + ρ1f * buf.Dt_buf + ρ2f * (buf.w - buf.u2)
        cg_solve!(x, buf.rhs, k, k_flip, ρ1f, ρ2f, buf;
                  maxit = cg_maxit, tol = cg_tol)

        # --- z-subproblem: TV (or Huber-TV) prox on Dx + u1 -----------------
        buf.zx_prev .= buf.zx
        buf.zy_prev .= buf.zy
        grad!(buf.Dx, buf.Dy, x)
        @. buf.cg_tmp3 = buf.Dx + buf.u1x
        @. buf.cg_tmp4 = buf.Dy + buf.u1y
        tv_prox!(buf.zx, buf.zy, buf.cg_tmp3, buf.cg_tmp4, λf / ρ1f, huber_δ_f)

        # --- w-subproblem: projection onto [0, clamp_hi] --------------------
        buf.w_prev .= buf.w
        @. buf.w = clamp(x + buf.u2, 0.0, clamp_hi_f)

        # --- dual (scaled) updates ------------------------------------------
        @. buf.u1x += buf.Dx - buf.zx
        @. buf.u1y += buf.Dy - buf.zy
        @. buf.u2  += x - buf.w

        # --- residuals ------------------------------------------------------
        # primal  r = (Dx−z, x−w)
        pri_norm = sqrt(sum(abs2, buf.Dx .- buf.zx) +
                        sum(abs2, buf.Dy .- buf.zy) +
                        sum(abs2, x .- buf.w))
        # dual    s = ρ₁ D^T (z−z_prev) + ρ₂ (w−w_prev)
        @. buf.cg_tmp3 = buf.zx - buf.zx_prev
        @. buf.cg_tmp4 = buf.zy - buf.zy_prev
        Dt!(buf.Dt_buf, buf.cg_tmp3, buf.cg_tmp4)
        dua_norm = sqrt(ρ1f^2 * sum(abs2, buf.Dt_buf) +
                        ρ2f^2 * sum(abs2, buf.w .- buf.w_prev))

        # Boyd-style tolerances
        nDx = sqrt(sum(abs2, buf.Dx) + sum(abs2, buf.Dy))
        nz  = sqrt(sum(abs2, buf.zx) + sum(abs2, buf.zy))
        nx  = norm(x)
        nw  = norm(buf.w)
        eps_pri = sqrt_p * abs_tol + rel_tol * max(nDx, nz, nx, nw)

        Dt!(buf.Dt_buf, buf.u1x, buf.u1y)          # rho_dual = ‖ρ₁ D^T u1 + ρ₂ u2‖
        rho_dual = sqrt(sum(abs2, ρ1f .* buf.Dt_buf .+ ρ2f .* buf.u2))
        eps_dual = sqrt_n * abs_tol + rel_tol * rho_dual

        # objective + relative x change
        obj_now  = objective(x, y, k_flip, λf, huber_δ_f, buf)
        dx_norm  = norm(x .- buf.x_prev)
        x_norm   = max(norm(x), 1e-30)
        rel_x    = dx_norm / x_norm
        push!(obj_hist,  obj_now)
        push!(pri_hist,  pri_norm)
        push!(dua_hist,  dua_norm)
        push!(relx_hist, rel_x)

        if verbose
            @printf("ADMM %3d  obj=%.6e  pri=%.3e/%.3e  dua=%.3e/%.3e  rel‖Δx‖=%.3e\n",
                    it, obj_now, pri_norm, eps_pri, dua_norm, eps_dual, rel_x)
        end

        # Stopping
        resid_ok = (pri_norm <= eps_pri) && (dua_norm <= eps_dual)
        obj_drop = abs(prev_obj - obj_now) / max(abs(obj_now), 1e-30)
        obj_ok   = it > 5 && obj_drop < obj_tol
        relx_ok  = xchange_tol > 0 && it > 1 && rel_x < xchange_tol
        if resid_ok
            converged = true; reason = :primal_dual; break
        elseif obj_ok
            converged = true; reason = :objective; break
        elseif relx_ok
            converged = true; reason = :x_change; break
        end
        prev_obj = obj_now
    end

    info = LocalTVInfo(iters_done, converged, reason,
                       obj_hist, pri_hist, dua_hist, relx_hist)
    return x, info
end

end 