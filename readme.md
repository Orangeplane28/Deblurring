# Deblurring.jl

Research and experimentation repository for image deblurring algorithms in Julia.

Current work focuses on:
- blind image deconvolution
- non-blind deconvolution
- variational optimization methods
- Total Variation priors
- FFT-based image restoration
- benchmarking and optimization of iterative methods

---

# Motivation

Satellite and large-scale visual data are heavily affected by blur, noise, and imperfect acquisition processes. These degradations can significantly affect downstream measurements, prediction systems, and visual analysis.

This project studies mathematical and optimization-based approaches for recovering sharp images from blurred observations.

The repository was developed as part of ongoing study into:
- inverse problems
- variational methods
- numerical optimization
- image restoration
- large visual data processing

---

# Current Implementations

## Wiener Deconvolution

Implemented:
- FFT-domain Wiener filtering
- RGB image support
- NSR regularization
- benchmarking experiments

Model:

```math
W(u,v)=\frac{K^*(u,v)}{|K(u,v)|^2+NSR}
```

---

## Blind Deblurring (PAM)

Blind deconvolution using:
- Projected Alternating Minimization (PAM)
- Total Variation image priors
- pyramid coarse-to-fine optimization
- iterative kernel estimation

Optimization model:

```math
\arg\min_{x,k} \|k*x-y\|_2^2 + \lambda J(x) + \gamma G(k)
```

Current work includes:
- parameter tuning
- convergence behavior
- kernel estimation improvements
- runtime optimization

---

## Hyper-Laplacian Refinement

Experimental implementation of Hyper-Laplacian non-blind refinement methods.

Current focus:
- iterative optimization
- quartic-root updates
- parameter sensitivity

Still under active development.

---

# Repository Structure

```text
.
├── src/
│   ├── Deblurring.jl
│   ├── pam.jl
│   ├── wiener.jl
│   └── hyper_laplacian.jl
├── experiment/
├── notes/
├── results/
├── Project.toml
└── Manifest.toml
```

- `src/` contains algorithm implementations
- `experiment/` contains experimentation and benchmarking scripts
- `notes/` contains derivations and research notes
- `results/` stores generated figures and outputs

---

# Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```


# Future Directions

Planned future directions:
- Optimization
- Blind ADMM methods
- improved kernel estimation
- large-scale visual data deconvolution
- patch-wise local PSF methods
- further Hyper-Laplacian refinement

---

# References

- Wiener filtering methods for image restoration
- Total Variation blind deconvolution methods
- MAP-based blind image deconvolution
- Hyper-Laplacian image priors