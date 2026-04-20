# Phase 2 — Calibration Module

> **Question 8**: *Use the data generated during the Phase 1 backtest to calibrate and refine the models.*

Phase 1 quoted EUR/USD on exchange A with **heuristic** `QuoterConfig` values (Avellaneda-Stoikov defaults, assumed volatility window, guessed risk aversion). This module replays the resulting fill history and mid-price path to replace those heuristics with statistically estimated values, then hands back a new `QuoterConfig` ready for a Phase 2 backtest.

**Scope — quoter only.** Client order flow (arrival `A`, decay `k_buy/k_sell`, Pareto size) is treated as an **exogenous market property** and is *not* calibrated. Keeping it fixed between Phase 1 and Phase 2 is what makes the P&L comparison meaningful.

---

## 1. Inputs from Phase 1

All inputs come from a completed `Controller`:

| Attribute | Type | Used by |
|---|---|---|
| `controller.trade_history` | DataFrame (one row per fill/hedge leg) | Spread, K, Stale, Gamma |
| `controller.step_log` | DataFrame (one row per step) | Spread, Stale |
| `controller.market_B.noised_mid_price` | ndarray | Volatility, Gamma |
| `controller.market_B.stock.time_step` | float (seconds) | all |
| `controller.quoter.capital_K` | float (EUR) | Spread |

Client-flow defaults (`A_BUY`, `K_BUY`, `ALPHA`, `SIZE_MIN`, ...) are read verbatim from [client_flow/config.py](../client_flow/config.py) and passed to `GammaOptimizer`.

---

## 2. Calibrators

Five independent modules, each exposing a `fit()` (or `optimize()`) method that returns a plain dict.

### 2.1 `volatility_calibrator.py` — `VolatilityCalibrator`

Calibrates **`vol_window_s`** (seconds; EWMA span used by the Quoter's realized-vol estimator).

- `fit_ewma()` — grid search `span ∈ {500, 1000, 2000, 4000, 6000, 10000}` steps, 70/30 train/test split on log-returns, pick the span minimizing out-of-sample RMSE of one-step variance prediction. Returns seconds, not steps.
- `fit_garch()` — optional GARCH(1,1) MLE (`scipy.optimize.minimize`, L-BFGS-B). Subsamples to 50k returns for speed. Returns `{alpha, beta, omega, persistence, aic, bic}` or `None`.
- `compare()` — runs both, reports AIC/BIC, returns `recommendation ∈ {"ewma", "garch"}`.

Reference: Bollerslev (1986); RiskMetrics (1996).

### 2.2 `spread_calibrator.py` — `SpreadCalibrator`

Calibrates **`alpha_spread`** and **`alpha_imbalance`** via OLS on MM fills:

```
|fill_price − fair_mid|  =  β₀  +  β₁ · (inventory / K)²  +  β₂ · imbalance  +  ε
```

where `imbalance` is a rolling (window=50) signed buy/sell ratio in [−1, 1]. Then:

- `alpha_spread = β₁ / β₀` (clipped to [0, 5])
- `alpha_imbalance = |β₂| / mean(fair_mid)` (clipped to [0, 0.01])

### 2.3 `k_calibrator.py` — `KCalibrator`

Calibrates **`k`** (arrival intensity decay, bp⁻¹). Since every full reprice posts all 10 ladder levels simultaneously, `n_posted` is ~constant across levels and fill counts are a proportional proxy for fill rate. Taking logs of `λ(δ) = A·exp(−k·δ)`:

```
log(n_fills[level])  =  const − k · mean_distance_bps[level]
```

OLS slope gives `k = −slope`. Run separately for buy/sell; `cfg.k` is the average (clipped to [0.01, 5.0]).

### 2.4 `stale_calibrator.py` — `StaleCalibrator`

Calibrates **`stale_s`** (seconds an order may rest before being considered stale). Models time-to-fill at level 1 as exponential with rate `λ₁ = n_fills_L1 / total_time`, then picks the 75th percentile:

```
stale_s  =  log(4) / λ₁     (clipped to [0.5, 60])
```

Level 1 is used because it has the highest `λ`, giving the tightest (lowest) threshold.

### 2.5 `gamma_optimizer.py` — `GammaOptimizer`

Calibrates **`gamma`** (risk aversion) and **`omega`** (infinite-horizon discount rate). Replicates the A-S half-spread formula from [quoter.py](../market_maker/quoter.py), then maximizes per-second expected net utility:

```
utility(γ, ω)  =  spread_pnl  −  inventory_risk_cost
             =  λ_total(s*) · s* · E[size]  −  γ · σ² · inv_horizon · λ_total · E[size]² / sec_per_year
```

Grid: `γ ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0}` × `ω ∈ {1/(24h), 1/(8h), 1/(4h)}`, refined around the argmax with `scipy.optimize.minimize_scalar` (bounded). Evaluates analytically on the historical path — does **not** re-run the simulation. Also returns an approximate `expected_sharpe` (daily).

Reference: Guéant, Lehalle & Fernandez-Tapia (2013).

---

## 3. Integration — `CalibratedConfigBuilder`

`calibrated_config.py` orchestrates the pipeline. It accepts a Phase 1 `Controller`, runs the five calibrators (parallelised with `ThreadPoolExecutor` where dependencies allow), and returns a single calibrated `QuoterConfig`.

```python
from src.utils.calibration import CalibratedConfigBuilder

builder    = CalibratedConfigBuilder(phase1_controller)
quoter_cfg = builder.build(target_sweep=0.05, beta=1.0)
print(builder.summary())
```

### User-tunable design inputs to `build()`

These are **not calibrated from data** — they are explicit design choices for the Phase 2 ladder:

| Input | Default | Meaning |
|---|---|---|
| `target_sweep` | `0.05` | Fraction of client orders expected to sweep past level 1. `Q_base = size_min · target_sweep^(−1/α)` via Pareto CDF inversion. At the default, `Q_base ≈ 7 370 EUR`. |
| `beta` | `1.0` | Ladder size decay per level. Not recoverable from fill volumes, so left as a manual knob. Default → level 2 carries ~37% of level 1's size. |

### Execution order

```
Round 1 (parallel, 4 workers):
    VolatilityCalibrator.fit_ewma()
    SpreadCalibrator.fit()
    KCalibrator.fit()
    StaleCalibrator.fit()

Q_base = size_min · target_sweep^(−1/α)          # closed form, no calibrator

Round 2 (parallel, 2 workers):
    GammaOptimizer.optimize()     # needs vol_params from Round 1
    VolatilityCalibrator.compare()
```

### Outputs

All calibrator dicts are kept under `builder.diagnostics` for inspection. The returned `QuoterConfig` overrides:

| Field | Source |
|---|---|
| `gamma`, `omega` | `GammaOptimizer` |
| `vol_window_s` | `VolatilityCalibrator` |
| `alpha_spread`, `alpha_imbalance` | `SpreadCalibrator` |
| `k` | `KCalibrator` |
| `stale_s` | `StaleCalibrator` |
| `Q_base`, `beta` | Pareto CDF inversion / user input |

All other `QuoterConfig` fields (`n_levels`, `tick_size`, latencies, fees, `weight_B/C`, `delta_limit`, `imbalance_window`, session multipliers, …) are inherited from defaults.

`builder.summary()` prints a styled heuristic-vs-calibrated table plus the expected daily Sharpe and the EWMA-vs-GARCH recommendation.

---

## 4. Phase 2 backtest

Rebuild a new `Controller` using the calibrated `QuoterConfig` and the **same** `ClientFlowConfig` as Phase 1, then call `simulate()` / `report()`. Compare P&L, Sharpe, fill rates, and inventory risk against Phase 1.

---

## 5. References

- **Avellaneda, M. & Stoikov, S.** (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
- **Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J.** (2013). *Dealing with the inventory risk: a solution to the market making problem.* Mathematics and Financial Economics, 7(4), 477–507.
- **Bollerslev, T.** (1986). *Generalized autoregressive conditional heteroskedasticity.* Journal of Econometrics, 31(3), 307–327.
- **Cartea, Á. & Jaimungal, S.** (2015). *Risk metrics and fine tuning of high-frequency trading strategies.* Mathematical Finance, 25(3), 576–611.
