# Phase 2 — Calibration Module

> **Question 8**: *Use the data generated during the Phase 1 backtest to calibrate and refine the models.*

Phase 1 quoted EUR/USD on exchange A using **heuristic parameters** (Avellaneda-Stoikov defaults, assumed Poisson arrival rates, guessed risk aversion). After one month of simulated trading, the `Controller` has accumulated fill history, per-step price/inventory snapshots, and market B/C price arrays. This module uses that data to **replace heuristics with statistically estimated parameters** and then re-runs a Phase 2 backtest with the calibrated configs.

---

## 1. Available Phase 1 Data

All data comes from a completed Phase 1 `Controller` instance.

### 1.1 Trade history — `Controller.trade_history` (DataFrame)

Delegates to `Quoter.trade_history`. Each row is a fill (MM on exchange A) or a hedge leg (on B/C):

| Column | Description |
|--------|-------------|
| `step`, `t` | Simulation step and elapsed seconds |
| `direction` | `"buy"` or `"sell"` (MM perspective) |
| `price` | Fill price |
| `size` | Filled quantity (EUR) |
| `fair_mid` | Weighted mid from B/C at fill time |
| `level` | Ladder level (1–10) for MM fills |
| `is_full_fill` | Whether the order was fully consumed |
| `inventory_after` | Inventory position after this fill |
| `fee_cost`, `cash_flow` | Fee and net cash impact |
| `is_hedge` | `True` for hedge legs on B/C |
| `venue` | `"A"`, `"B"`, or `"C"` |

### 1.2 Step log — `Controller.step_log` (DataFrame)

One row per simulation step, logged by `Controller._log_step()`:

| Column | Description |
|--------|-------------|
| `step`, `t` | Simulation step and elapsed seconds |
| `bid_A`, `ask_A`, `mid_A` | Best resting MM quotes on exchange A |
| `bid_B`, `ask_B`, `mid_B` | Market B prices at this step |
| `bid_C`, `ask_C`, `mid_C` | Market C prices at this step |
| `fair_mid` | Weighted B/C mid (`weight_B * mid_B + weight_C * mid_C`) |
| `inventory` | Current EUR inventory |
| `n_mm_resting` | Number of resting MM orders on A |
| `fills_this_step` | New fills since the previous step |

This replaces any notion of a separate "quote history" — the step_log contains the quote distances needed for arrival calibration: at each step, `|bid_A - fair_mid|` and `|ask_A - fair_mid|` give the distance from mid to our active quotes.

### 1.3 Market price arrays

From `Market` objects for B and C (accessible via `Controller.market_B`, `Controller.market_C`):
- `market.noised_mid_price` — mid-price at every simulation step
- `market.bid_price`, `market.ask_price` — best bid/ask at every step

### 1.4 P&L report — `Controller.pnl_report()`

Summary dict from `PnLTracker.report()` including total MtM P&L, realized P&L, fees, etc. Used as the baseline to measure improvement.

---

## 2. Calibration Modules

> **Design principle:** Only parameters that control the MM's *quoting strategy* are calibrated. Client order flow (arrival rates, size distribution) is an exogenous market property — changing it between Phase 1 and Phase 2 would mean comparing against a different market, not a better strategy. Default arrival/size values from `client_flow/config.py` are passed as fixed inputs to the GammaOptimizer.

### 2.1 `volatility_calibrator.py` — Volatility Model Selection

**What it calibrates:** `vol_window` (in `QuoterConfig`), optionally GARCH(1,1) parameters

**Phase 1 default:** `vol_window=6000` (EWMA span for realized vol)

**Methods:**

1. **EWMA span optimization:**
   - The quoter computes sigma via `pd.Series.ewm(span=vol_window).var()`
   - Optimal span minimizes squared prediction error of realized variance over the next N steps
   - Grid search over `span in [500, 1000, 2000, 4000, 6000, 10000]`, evaluate out-of-sample RMSE

2. **GARCH(1,1) MLE:**
   - Fit `sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2` to log-returns from market B
   - MLE via `scipy.optimize.minimize` on the conditionally Gaussian log-likelihood
   - The existing `generate_garch_path` in `stock_simulation` already uses these parameters; calibrating them from data closes the loop

3. **Model selection:** Compare EWMA vs GARCH via AIC/BIC

**References:** Bollerslev (1986); RiskMetrics (1996) for EWMA

**Interface:**
```python
class VolatilityCalibrator:
    def __init__(self, mid_prices: np.ndarray, dt: float)
    def fit_ewma(self) -> dict         # {"vol_window": int, "rmse": float}
    def fit_garch(self) -> dict        # {"alpha", "beta", "omega", "persistence", "aic"}
    def compare_models(self) -> str    # "ewma" or "garch" with AIC/BIC table
```

---

### 2.2 `spread_calibrator.py` — Spread Component Calibration

**What it calibrates:** `alpha_spread` (inventory spread weight), `alpha_imbalance` (order flow imbalance tilt) in `QuoterConfig`

**Phase 1 defaults:** `alpha_spread=0.5`, `alpha_imbalance=0.0002`

**Method — Regression on realized spreads:**

From the step_log, compute the realized half-spread `(ask_A - bid_A) / 2` and the corresponding inventory ratio and imbalance at each step. Regress the effective spread earned on fills against its components:

```
effective_spread_i = beta_0 + beta_1 * inventory_ratio_i^2 + beta_2 * imbalance_i + eps_i
```

where `effective_spread_i = |fill_price - fair_mid|` for each MM fill.

The coefficients map to:
- `beta_1` → `alpha_spread` (how much inventory state widens the spread)
- `beta_2` → `alpha_imbalance` (how much order flow tilts the reservation price)

**Interface:**
```python
class SpreadCalibrator:
    def __init__(self, fill_df: pd.DataFrame, step_log: pd.DataFrame)
    def fit(self) -> dict    # {"alpha_spread", "alpha_imbalance"}
```

---

### 2.3 `gamma_optimizer.py` — Risk Aversion & Horizon Parameters

**What it calibrates:** `gamma` and `omega` (in `QuoterConfig`)

**Phase 1 defaults:** `gamma=0.1`, `omega=1/(8*3600)` (≈ one FX session horizon)

**Why this matters:** Gamma controls the trade-off between spread revenue and inventory risk in the A-S reservation price:
```
r(s, q, t) = s - q * gamma * sigma^2 * (1 / (omega * T_year)) * penalty_factor
```
Too low → accumulates inventory risk. Too high → quotes too wide, misses fills.

Omega controls how far into the future inventory risk is penalized: smaller omega = longer horizon = heavier penalty.

**Method — Grid optimization on expected net utility:**

For each candidate `(gamma, omega)`, at each historical step t:
1. Compute the theoretical half-spread from the A-S formula
2. Compute expected fill rate using calibrated `A, k` from `ArrivalCalibrator`
3. Expected spread P&L per unit time: `2 * lambda(s) * s * E[size]`
4. Expected inventory variance cost: `gamma * sigma^2 * Var(inventory)`
5. Net expected utility = spread P&L - inventory cost

Grid search over `gamma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]` × `omega in [1/(24*3600), 1/(8*3600), 1/(4*3600)]`, refine with golden-section search around the optimum. This evaluates **analytically on the historical path** — it does not re-run the simulation.

**References:** Gueant, Lehalle & Fernandez-Tapia (2013), Section on optimal control parameter selection

**Interface:**
```python
class GammaOptimizer:
    def __init__(self, fill_df: pd.DataFrame, step_log: pd.DataFrame,
                 mid_prices: np.ndarray, sigma: np.ndarray, dt: float,
                 arrival_params: dict, size_params: dict)
    def optimize(self) -> dict   # {"gamma": float, "omega": float, "expected_sharpe": float}
```

---

## 3. Integration: `calibrated_config.py`

Orchestrates the full calibration pipeline and outputs config objects that the existing `Quoter` and `ClientFlowGenerator` consume directly — **no changes needed to Phase 1 code**. The calibrated configs are then passed to a new `Controller` for the Phase 2 backtest.

```python
class CalibratedConfigBuilder:
    def __init__(self, phase1_controller: Controller)
    def calibrate_all(self) -> dict         # Runs all calibrators, returns summary
    def build_quoter_config(self) -> QuoterConfig
    def build_client_flow_config(self) -> ClientFlowConfig
    def summary_report(self) -> str         # Heuristic vs calibrated comparison table
```

**Data flow:**
```
Phase 1: Controller.simulate()
    │
    ├── Controller.trade_history  (DataFrame — all fills)
    ├── Controller.step_log       (DataFrame — per-step quotes, prices, inventory)
    ├── Controller.market_B / market_C  (price arrays)
    └── Controller.pnl_report()   (Phase 1 P&L baseline)
          │
          ▼
  CalibratedConfigBuilder(phase1_controller)
  CalibratedConfigBuilder.calibrate_all()
          │
          ├── ArrivalCalibrator.fit()     → A_buy, A_sell, k_buy, k_sell
          ├── SizeCalibrator.fit()        → alpha, size_min, size_max
          ├── VolatilityCalibrator.fit()  → vol_window
          ├── SpreadCalibrator.fit()      → alpha_spread, alpha_imbalance
          └── GammaOptimizer.optimize()   → gamma, omega  (uses results from above)
          │
          ▼
  QuoterConfig(gamma=calibrated, k=calibrated, omega=calibrated,
               vol_window=calibrated, alpha_spread=calibrated,
               alpha_imbalance=calibrated, ...)
  ClientFlowConfig(A_buy=calibrated, k_buy=calibrated,
                   alpha=calibrated, ...)
          │
          ▼
  Phase 2: new Controller(market_B2, market_C2, book2,
                          Quoter(..., config=calibrated_quoter_cfg),
                          ClientFlowGenerator(config=calibrated_flow_cfg).generate_step)
  Controller.simulate()  →  Controller.report()
          │
          ▼
  Compare Phase 1 vs Phase 2 P&L to validate calibration
```

---

## 4. Parameter Mapping

| Parameter | Phase 1 Heuristic | Calibrator | Config Class | Notes |
|-----------|------------------|------------|--------------|-------|
| `gamma` | 0.1 | GammaOptimizer | `QuoterConfig` | Most impactful for P&L |
| `omega` | 1/(8×3600) | GammaOptimizer | `QuoterConfig` | Infinite-horizon discount rate |
| `vol_window` | 6000 | VolatilityCalibrator | `QuoterConfig` | EWMA span |
| `alpha_spread` | 0.5 | SpreadCalibrator | `QuoterConfig` | Inventory spread weight |
| `alpha_imbalance` | 0.0002 | SpreadCalibrator | `QuoterConfig` | Order flow imbalance tilt |

**Not calibrated — exogenous market (identical between Phase 1 and Phase 2):**
Client order flow parameters (`A_buy`, `A_sell`, `k_buy`, `k_sell`, `alpha`, `size_min`, `size_max`) are properties of the market, not our strategy. Changing them between phases would mean comparing against a different market environment, making the P&L comparison meaningless. Default values from `client_flow/config.py` are used as fixed inputs to the GammaOptimizer.

**Not calibrated — pass-through from Phase 1:** `k` (QuoterConfig), `n_levels`, `beta`, `Q_base`, `tick_size`, latencies, fee structure, `weight_B/C`, `delta_limit`, `market_order_ratio`, `use_asymmetric_delta`, `imbalance_window`, `stale_steps`, `requote_threshold_spread_fraction`.

---

## 5. Implementation Order

```
1. volatility_calibrator.py    (independent)

2. spread_calibrator.py        (independent)

3. gamma_optimizer.py          (uses default arrival/size + calibrated vol)

4. calibrated_config.py        (integration layer, depends on all three)
```

---

## 6. Phase 2 Backtest

After calibration, run a new 1-month backtest with calibrated parameters:

1. Generate new market B and C price paths (fresh month of data, same model)
2. Build calibrated `QuoterConfig` via `CalibratedConfigBuilder`
3. Instantiate `Quoter` with calibrated config, `ClientFlowGenerator` with **default** config (same as Phase 1)
4. Run `Controller.simulate()` and `Controller.report()`
5. Compare Phase 1 vs Phase 2: total P&L, Sharpe ratio, fill rates, inventory risk

The Controller's `report()` method already produces all the plots and metrics required by Question 7 — the same report is generated for both phases, enabling direct comparison.

---

## 7. Academic References

- **Avellaneda, M. & Stoikov, S.** (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217-224.
- **Gueant, O., Lehalle, C.-A. & Fernandez-Tapia, J.** (2013). *Dealing with the inventory risk: a solution to the market making problem.* Mathematics and Financial Economics, 7(4), 477-507.
- **Aban, I.B., Meerschaert, M.M. & Panorska, A.K.** (2006). *Parameter estimation for the truncated Pareto distribution.* Journal of the American Statistical Association, 101(473), 270-277.
- **Gabaix, X., Gopikrishnan, P., Plerou, V. & Stanley, H.E.** (2003). *A theory of power-law distributions in financial market fluctuations.* Nature, 423, 267-270.
- **Bollerslev, T.** (1986). *Generalized autoregressive conditional heteroskedasticity.* Journal of Econometrics, 31(3), 307-327.
- **Cartea, A. & Jaimungal, S.** (2015). *Risk metrics and fine tuning of high-frequency trading strategies.* Mathematical Finance, 25(3), 576-611.
