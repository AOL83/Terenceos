# TerenceOS

**Repository name:** TerenceOS  
**Description:** Local independence of statistical invariants in an open delayed expanding system.

TerenceOS is a deterministic, reproducible computational pipeline for studying an open delayed expanding dynamical system and estimating the local Jacobian of behavioral invariants.

## Doctrine (authoritative text)

```text
One time unit equals one application of the delayed map step: (x_t, y_t) -> (x_{t+1}, y_{t+1}).

Escape is decided using the newly produced x-coordinate x_{t+1}: a trajectory is removed at step t when circle_distance(x_{t+1}, a) < kappa.

Ulam orientation is row-substochastic: rows index source cells and columns index destination cells; each row is normalized by the number of deterministic samples started in that source cell, yielding row sums <= 1 in the open system.
```

`DOCTRINE_HASH = 19a8ebd5039b`

## Mathematical definition

For parameters \(p=(\theta,\mu,\kappa,a)\), the circle map is

\[
f_\theta(x) = \left(2x + \theta \sin(2\pi x)\right) \bmod 1.
\]

The delayed lifted update is

\[
x_{t+1} = \left((1-\mu)f_\theta(x_t) + \mu f_\theta(y_t)\right) \bmod 1,
\qquad y_{t+1}=x_t.
\]

Hole on the circle:

\[
H(a,\kappa)=\left\{x\in\mathbb T^1:\ d_\circ(x,a)<\kappa\right\},
\quad d_\circ(x,a)=\min\{|x-a|,1-|x-a|\}.
\]

Escape convention: a trajectory escapes at step \(t\) when \(x_{t+1}\in H(a,\kappa)\).

Jacobian used for tangent iteration:

\[
J(x,y)=
\begin{bmatrix}
(1-\mu)f_\theta'(x) & \mu f_\theta'(y) \\
1 & 0
\end{bmatrix},
\quad
f_\theta'(x)=2+\theta 2\pi\cos(2\pi x).
\]

Behavioral map:

\[
B(\theta,\kappa,\mu)=\big(\lambda_{\max},\rho,-\log|\lambda_2|\big),
\]

where \(\lambda_{\max}\) is the conditioned Lyapunov exponent, \(\rho\) the escape rate, and \(\lambda_2\) the subleading Ulam eigenvalue.

Conjecture target:

\[
\det DB(p_0)\neq 0.
\]

Time unit convention: one map step equals one unit time.  
Ulam orientation: source cells are rows, destination cells are columns, producing a row-substochastic open transfer matrix.

Plateau certification: finite differences are evaluated over descending \(h\)-values; per parameter direction, the first consecutive pair with relative change below tolerance and seed-wise sign consistency defines the certified plateau index \(h^*\).

## Minimal flow diagram

```text
SystemParameters -> DelayedOpenExpandingSystem -> behavioral_map
    -> (Lyapunov, Escape, Ulam) -> finite_difference_jacobian -> report bundle
```

## User manual

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the pilot

```bash
python -m src.run_pilot --config configs/pilot.yaml
```

The command prints the created bundle directory: `results/run_<timestamp>/`.

### Output interpretation

- `jacobian_summary.json`
  - `J_mean`, `J_sd`: seed-aggregated Jacobian mean/std.
  - `det_distribution`, `det_ci95`: determinant distribution and 95% CI.
  - `singular_values_distribution`, `singular_values_ci95`: singular-value uncertainty.
  - `cond_distribution`, `cond_mean`: conditioning diagnostics.
  - `column_normalized_det_distribution`, `column_normalized_det_ci95`: scale-normalized determinant diagnostics.
  - `derivative_by_h`: mean/sd finite differences vs step size for each direction.
- `plateau_info.json`
  - per direction logs of relative-change checks and selected `chosen_h`.
- `ulam_logs.json`
  - sparse matrix dimensions, `nnz`, eigenvalue metadata, convergence warnings.
- `escape_logs.json`
  - fitted slope/intercept, `rho`, window policy, and candidate-window count.
- `tables/grid_refinement.csv`
  - refinement of `M` with `lambda1` and proxy `-log(lambda1)`.

### Bundle required files checklist

- `meta.json`
- `doctrine.txt`
- `jacobian_summary.json`
- `plateau_info.json`
- `ulam_logs.json`
- `escape_logs.json`
- `tables/grid_refinement.csv`
- `plots/plateau_theta.png`
- `plots/plateau_kappa.png`
- `plots/plateau_mu.png`
- `plots/survival_fit.png`

### Reproducibility

All determinism-critical seeds are stored in `meta.json`:
- `seeds.Z0_seed`
- `seeds.replicate_seeds`

Finite-difference evaluations use a common `Z0` across all perturbations and all replicate seeds.
