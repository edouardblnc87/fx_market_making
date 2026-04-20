# stock_simulation — True Asset Value Path

## Context

This is the root of the simulation pipeline: every downstream module (venues B/C, order book, quoter, client flow) anchors to the price path produced here. The `Stock` class generates a single "true" underlying value process `V[t]` that represents the latent fair price of EUR/USD, **before** any venue-specific microstructure noise is added.

Three models are supported, all simulated on a second-based time grid:

| Model | When to use |
|---|---|
| **GBM** | Default benchmark — constant vol, fast, analytically tractable. |
| **GARCH(1,1) + Merton jumps** | Realistic FX intraday — vol clustering (α+β near 1) with optional discrete jumps. |
| **Heston** | Continuous stochastic vol with leverage (ρ) — Feller-condition-aware. |

Venues B and C wrap **the same `Stock` instance** (deep-copied) so the underlying fair value is identical — only independent microstructure noise and spread regimes differentiate them. See [market_simulator/README.md](../market_simulator/README.md) for the downstream stage.

---

## Pipeline

```
Stock(drift, vol, origin, tick_size)
    │
    ▼
simulate_gbm(n_days, dt_seconds)            ← or simulate_garch / simulate_heston
    │  stock.simulation  V[t]       shape (N+1,)
    │  stock._time_grid  seconds    shape (N+1,)
    │  stock.time_step   dt (s)
    │  stock.n_steps     N
    │  stock.vol_realized            per-step innovations (GARCH/Heston: diffusion only)
    │  stock.vol_path   (Heston/GARCH only) annualised instantaneous vol
    ▼
Market(stock)  →  generate_noised_mid_price(...)  →  build_spread(...)
```

Time conventions live in [config.py](config.py):

- `TRADING_SECONDS_PER_DAY  = 24 × 3600` (FX is 24/5).
- `TRADING_SECONDS_PER_YEAR = 252 × TRADING_SECONDS_PER_DAY ≈ 7.26M s`.

`generate_time_grid(n_days, dt_seconds)` returns the shared `np.linspace(0, T, N+1)` used by all three simulators.

---

## Models

### GBM — `simulate_gbm(n_days=30, dt_seconds=0.05)`

Default: 30 days at 50ms step ≈ 51.8M steps. Implemented in [stochastic_simulation.py:generate_gbm_path](stochastic_simulation.py).

```
dS = μ S dt + σ S dW

σ_dt = σ_ann · √(dt / T_year)
μ_dt = (μ_ann − ½σ_ann²) · dt / T_year     ← Itô correction built in
log_S[t+1] = log_S[t] + μ_dt + σ_dt · Z[t]    Z ~ 𝒩(0, 1)
```

Vectorised — one pass over the whole path via `cumsum`. Constant per-step variance; no fat tails, no clustering.

### GARCH(1,1) + Merton Jumps — `simulate_garch(n_days, dt_seconds, alpha, beta, lam, mu_J, sigma_J)`

Stationarity check raises if `α + β ≥ 1`. Default `α=0.05, β=0.94` (FX intraday convention, persistence 0.99).

```
h[t+1] = ω + α·ε[t]² + β·h[t],   ε[t] = √h[t]·Z[t],   ω = σ²·dt_year·(1 − α − β)
N[t]   ~ Bernoulli(λ·dt_year),   J[t] ~ 𝒩(μ_J, σ_J²)
r[t]   = μ·dt_year − ½·h[t] + ε[t] + N[t]·J[t]
```

**Jump separation:** jumps are additive to the log-return but do **not** feed the GARCH recursion (`h[t+1]` depends only on the diffusion innovation `ε`). Standard convention — clustering reflects normal moves; jumps are exogenous events. `lam=0` disables jumps entirely. `lam = 252·3 ≈ 756` gives ~3 jumps/day.

Sequential loop — cannot be vectorised because `h[t+1]` depends on `ε[t]`.

### Heston — `simulate_heston(n_days, dt_seconds, v0, kappa, theta, xi, rho)`

Full-truncation Euler-Maruyama discretisation (Lord et al. 2010). Defaults: `v0 = θ = σ²`, `κ=2.0`, `ξ=0.3`, `ρ=−0.1` (mild FX leverage).

```
dS = μ S dt + √v S dW_S
dv = κ(θ − v) dt + ξ √v dW_v,   corr(dW_S, dW_v) = ρ
```

Variance is generated via Cholesky: `Z_v = ρ·Z_S + √(1−ρ²)·Z_ind`. Both `v[t+1]` and the square-root coefficient are floored at zero (full truncation).

**Feller condition** `2κθ/ξ² > 1` is checked; a `RuntimeWarning` fires if violated (variance can hit zero and results degrade).

---

## Utility Methods

| Method | Returns |
|---|---|
| `compute_realized_volatility(window_size)` | Annualised rolling RV (zero-mean estimator, O(N) via cumsum). Used by the spread builders in `market_simulator`. |
| `plot_path()` | Dark-theme single-panel price chart with model-tagged title (GBM / Heston params). |
| `plot_vol_path()` | 2-panel chart: price + instantaneous annualised vol. GARCH or Heston only. |
| `sanity_check()` | Prints realised vol / drift / skew / kurtosis against expected values. For GARCH/Heston: aggregates returns into ~60s buckets (exposes fat tails hidden at per-step scale by CLT) and prints model-specific diagnostics (Feller, persistence, vol-of-vol, autocorr of squared returns). |

The sanity-check output is the first thing to look at when a path "looks wrong" — realised vs. expected is the contract between the parameter and the generator.

---

## Key Invariants

- **Parameters are annualised; the simulator scales them.** `σ` and `μ` are always passed in annualised terms (market convention). Never pre-scale them to the step.
- **Itô correction is included.** `μ_dt = (μ − ½σ²)·dt/T_year`. Passing a risk-neutral drift means passing `drift = r` (not `r − ½σ²`).
- **`simulation` is a 1D array of length `N+1`.** Index 0 is `S0`. The `_time_grid` has the same length — both aligned.
- **`vol_realized` length is `N`, not `N+1`** (per-step innovations, one per interval). For GARCH/Heston this is the **diffusion** part only; jumps (GARCH) are not included in `vol_realized`.
- **GBM vs. stochastic-vol paths share attribute names** (`simulation`, `time_step`, `n_steps`, `vol_realized`). `variance_path` and `vol_path` exist only after GARCH or Heston. `self.sim_type ∈ {'gbm', 'garch', 'heston'}` tags the active model.
- **Tick-size snapping is optional.** Pass `tick_size=0` to disable — useful when the downstream code expects the raw float path (e.g. the realised-vol estimator).
- **Feller check is a warning, not a hard error.** The Heston path still runs; results may be unreliable near the zero-variance boundary.

---

## Module Map

| File | Responsibility |
|---|---|
| [stock.py](stock.py) | `Stock` class — wraps the three generators, stores state, provides plotting and sanity checks. |
| [stochastic_simulation.py](stochastic_simulation.py) | Pure functions: `generate_time_grid`, `generate_gbm_path`, `generate_garch_path`, `generate_heston_path`. No class state. |
| [config.py](config.py) | Time constants (`TRADING_SECONDS_PER_DAY`, `TRADING_SECONDS_PER_YEAR`). |

---

## Usage

```python
from utils.stock_simulation import Stock

stock = Stock(drift=0.0, vol=0.08, origin=1.10, tick_size=1e-5)
stock.simulate_garch(n_days=5, dt_seconds=0.05, alpha=0.05, beta=0.94,
                     lam=756, mu_J=0.0, sigma_J=0.005)   # ~3 jumps/day
stock.sanity_check()
stock.plot_vol_path()

# Downstream: wrap in Market for venues B and C
from utils.market_simulator import Market
import copy
market_B = Market(stock)
market_C = Market(copy.deepcopy(stock))   # independent noise draw
```