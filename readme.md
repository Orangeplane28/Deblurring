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

## Motivation

Satellite data provides information that cannot be effectively obtained from the ground.

Governments and companies invest heavily in satellites to continuously monitor large regions of Earth, and many prediction systems and large-scale measurements depend directly on this visual data.

One important application is measuring destruction in cities affected by natural disasters. Accurate visual measurements help guide economic investment, urban planning, and resource allocation.

However, satellite imagery is often degraded by:
- blur
- noise
- imperfect acquisition processes

These degradations can significantly affect downstream measurements and predictions.

For example, older satellite systems with blurrier imagery caused nearby regions to appear more similar than they actually were, leading to substantially inflated damage estimates in typhoon-impact studies.

This project studies mathematical and optimization-based approaches for recovering sharp images from blurred observations through:
- inverse problems
- variational optimization
- image priors
- blind deconvolution
- non-blind deconvolution
- large visual data restoration

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
X(u,v)=
\frac{K^*(u,v)}
{|K(u,v)|^2 + NSR}
Y(u,v)
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

### Vectorial Total Variation Prior

The blind deconvolution implementation uses a Vectorial Total Variation (VTV) prior for RGB images, coupling all color channels into a single regularization term.

```math
J(x)=\sum_{i,j}\sqrt{
\sum_{c\in\{r,g,b\}}
\left(
|\nabla_x x^c_{i,j}|^2 +
|\nabla_y x^c_{i,j}|^2
\right)
}
```

where:
- `∇x` is the horizontal image gradient
- `∇y` is the vertical image gradient
- `c ∈ {r,g,b}` indexes the RGB channels

The corresponding divergence term is:

```math
\nabla J(x)=div\left(
\frac{\nabla x}{|\nabla x|}
\right)
```

Unlike channel-wise TV, Vectorial TV couples RGB gradients together, helping preserve aligned color edges and reducing color artifacts during deconvolution.

---

### Hyper-Laplacian Prior

The repository also contains experiments with Hyper-Laplacian image priors for sparse image-gradient regularization.

Natural image gradients are commonly modeled with a heavy-tailed distribution:

```math
p(\nabla x)\propto e^{-|\nabla x|^\alpha}
```

with:

```math
0 < \alpha < 1
```

The current implementation focuses on:

```math
\alpha=\frac{2}{3}
```

leading to the regularization term:

```math
J(x)=\sum_{i,j}|\nabla x_{i,j}|^{2/3}
```

This prior preserves strong edges while suppressing weaker noise-like gradients.

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