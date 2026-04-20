# market_simulator — Reference Venues B and C

## Context

Exchange A (our venue) has no trading history in Phase 1, so all quoting logic must anchor to external reference prices. Two public venues provide that anchor:

- **Exchange B** — 75% of EUR/USD volume, 200ms feed latency
- **Exchange C** — 25% of EUR/USD volume, 170ms feed latency

The `Market` class wraps a `Stock` (GBM / GARCH / Heston path) into one of these venues by adding microstructure noise, a bid/ask spread, and a quoted-size series. Venues B and C share the same underlying `Stock` via `copy.deepcopy` so their true value is identical — only the independent noise realisations and (optionally) different spread regimes differentiate them.

Downstream, the [Quoter](../market_maker/README.md) reads `market.bid_price` / `market.ask_price` (properties resolving to whichever spread was last built) with latency offsets applied, then takes the cross-venue best bid/ask as its fair mid.

---

## Pipeline

```
Stock (GBM/GARCH/Heston)
    │  stock.simulation  V[t]
    ▼
generate_noised_mid_price(vol_factor)
    │  noised_mid_price = V + ε,   ε ~ 𝒩(0, (vol_factor·σ)²·dt)
    ▼
build_spread(option=...)            ← one of 5 regimes, see below
    │  bid_price / ask_price arrays (active regime exposed via @property)
    ▼
generate_depth(mean_eur)            ← optional, inverse-proportional to spread
       depth[t] = mean_eur × mean_spread / spread[t]
```

All arrays are length `n_steps + 1`, aligned on `stock._time_grid`.

---

## Spread Regimes

Selected via `market.build_spread(option=..., **kwargs)`. The last regime built becomes the "active" one — `market.bid_price` / `market.ask_price` route to its arrays. Individual regime arrays are always preserved (`bid_price_constant`, `bid_price_sto`, …) so multiple regimes can be compared on the same path.

### Static — `build_static_spread(spread_bps=1.0)`

Constant half-spread in bps of mid. No vol input.

```
half_spread(t) = mid(t) × spread_bps / 10_000
```

Baseline for benchmarking; identical noise model as the others but no dynamics.

### Stochastic — `build_stochastic_spread(window_size, alpha, spread_bps)`

Half-spread scales linearly with annualised realised vol, measured over a rolling window.

```
half_spread(t) = h_0(t) × (1 + α × RV_ann(t) / σ)
```

Simple vol-adaptive benchmark. No floor, no asymmetry, no memory beyond the rolling window.

### Adaptive — `build_adaptive_spread(window_size, alpha, spread_bps)`

Variant of Stochastic that reacts only to **deviations** from parametric vol, with a 10%-of-baseline floor.

```
half_spread(t) = max( h_0(t) × (1 + α × (RV_ann(t)/σ − 1)),  0.1 × h_0(t) )
```

At normal vol (`RV = σ`), spread equals the static baseline. It widens above and tightens below — unlike Stochastic, which always adds `α·RV/σ` on top.

### Asymmetric — `build_asymmetric_spread(s0, alpha, kappa_u, kappa_d, window_size, spread_bps, sigma_s)`

The flagship model. A static floor `S_0` plus a mean-reverting excess spread `S_excess` that widens fast and tightens slowly.

```
half_spread(t) = S_0(t) + S_excess(t)
S_star(t)      = S_0(t) × α × RV_ann(t) / σ                (vol-driven target)
S_excess[t+1]  = (1 − κ·dt)·S_excess[t] + κ·dt·S_star[t]   κ = κ_u if widening else κ_d
                 floored at 0
```

Defaults `κ_u = 50/s` (half-life ~14ms), `κ_d = 2/s` (half-life ~350ms) encode adverse-selection fear vs. competitive pressure. See [theory/asymmetric_spread_model.md](theory/asymmetric_spread_model.md) for the full derivation and parameter calibration.

### Skewed — `build_skewed_spread(window_size, alpha, gamma, ema_span, threshold, spread_bps)`

Vol-adaptive width (same formula as Stochastic, capped at 5×σ) with a **directional centre shift** driven by EMA momentum. The centre shifts only when normalised momentum exceeds a dead-band threshold.

```
half_spread(t) = h_0(t) × (1 + α × min(RV_ann/σ, 5))
norm_mom(t)    = EMA(log-returns) / σ_step_ref
dead_band(t)   = sign(norm_mom) · max(|norm_mom| − threshold, 0)
skew(t)        = γ × half_spread(t) × tanh(dead_band(t))

ask(t) = mid(t) + half_spread(t) + skew(t)
bid(t) = mid(t) − half_spread(t) + skew(t)
```

Theory for an OFI-driven extension of this model lives in [theory/version3_OFI_model.md](theory/version3_OFI_model.md). Note: the OFI variant is **not implemented in `Market`** — it is referenced by the Quoter's own OFI tilt (see [market_maker/README.md](../market_maker/README.md) Step 4).

---

## Depth

`generate_depth(mean_eur=500_000)` produces a per-step EUR size quoted at the best price on the active bid/ask. Depth is inversely proportional to spread width:

```
depth[t] = mean_eur × mean_spread / spread[t]
```

Tight spread → confident market, larger quoted size. Wide spread → thin liquidity, smaller size. The variability comes entirely from the spread dynamics; no additional random process. Required by the hedge router, which caps each hedge leg at `depth[step − latency]` (same latency as the price feed).

---

## Noise Model

`generate_noised_mid_price(vol_factor=0.1)` adds i.i.d. Gaussian microstructure noise on top of the true GBM path:

```
ε[t]               ~ 𝒩(0, 1)
noised_mid[t]      = V[t] + vol_factor × σ × √(dt / T_year) × ε[t]
```

`vol_factor` scales the noise as a fraction of the annualised GBM vol. Default 0.1 = 10% of parametric σ per step. Noise is independent across venues B and C (different RNG draws for the same `Stock`).

Realised-vol estimation inside the spread builders uses `stock.simulation` (the true `V[t]`), **not** `noised_mid_price` — see [spread_utils.py:compute_rv_zero_mean](spread_utils.py) for why: the noised mid's vol is inflated by the microstructure variance and would bias the estimate upward.

---

## Key Invariants

- **Call order is fixed.** `generate_noised_mid_price` → `build_spread(...)` → `generate_depth(...)`. Each step consumes the previous one's output; skipping raises.
- **`bid_price` / `ask_price` are properties, not arrays.** They resolve via `_active_spread` (set by whichever builder ran last). The underlying per-regime arrays (`bid_price_constant`, `bid_price_sto`, …) remain available after a switch.
- **Zero-mean vol estimator by design.** Short windows (10–5000 steps) make the sample mean pure noise; subtracting it adds variance rather than removing drift. See [spread_utils.py](spread_utils.py) for the full rationale.
- **Stability of asymmetric evolution.** `κ × dt < 1` must hold or the discrete update overshoots. Defaults at `dt = 10ms` give `κ_u·dt = 0.5` (safe margin).
- **B and C share a `Stock`.** In notebooks: `market_B = Market(stock); market_C = Market(copy.deepcopy(stock))`. The deepcopy is mandatory — sharing would couple the noise paths.

---

## Module Map

| File | Responsibility |
|---|---|
| [market.py](market.py) | `Market` class — noise, 5 spread builders, depth, diagnostics. |
| [spread_utils.py](spread_utils.py) | `compute_rv_zero_mean`, `evolve_s_excess` — math helpers shared by multiple builders. |
| [theory/asymmetric_spread_model.md](theory/asymmetric_spread_model.md) | Full derivation of the asymmetric mean-reverting spread. |
| [theory/version3_OFI_model.md](theory/version3_OFI_model.md) | OFI-driven spread extension (theory only, not implemented in `Market`). |

---

## Diagnostics

- `plot_noised_mid_price(series=[...])` — overlay mid, bid/ask of selected regimes, and depth (EUR, right axis).
- `compare_spreads()` — one ask panel, one bid panel, one width panel per built regime (with RV overlay and `corr(width, RV)` in the title), plus a dedicated skew-δ panel if skewed was built.
- `plot_comparison()` — true GBM vs noised mid, with deviation residuals (mean / std printed).
- `sanity_check()` — realised noise std, noise vol annualised, total mid vol, noise-to-signal ratio.
- `sanity_check_spreads()` — per-regime spread stats (mean / std / min / max / relative %), ask/bid bias, quote-centre offset (flags skewed regimes).

---

## Reading Order for New Contributors

1. This README.
2. [theory/asymmetric_spread_model.md](theory/asymmetric_spread_model.md) before touching `build_asymmetric_spread`.
3. [theory/version3_OFI_model.md](theory/version3_OFI_model.md) if you plan to integrate OFI directly into the venue spread (currently OFI lives in the Quoter, not here).
4. [spread_utils.py](spread_utils.py) — short, worth reading end-to-end for the vol-estimator and asymmetric-evolution rationale.
