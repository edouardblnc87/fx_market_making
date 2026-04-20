# FX Market Making — EUR/USD Simulation

A full simulation of a market-maker (MM) operating on the EUR/USD pair, built as a Master's project at Université Paris Dauphine. The project models the complete MM lifecycle: price path generation, order book management, client order flow, P&L tracking, parameter calibration, and competition against HFT agents.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [How to Run](#4-how-to-run)
5. [Simulation Pipeline](#5-simulation-pipeline)
6. [Module Reference](#6-module-reference)
7. [Parameters Reference](#7-parameters-reference)

---

## 1. Project Overview

The simulation models the following setup:

- **Exchange A** — the MM's own venue. It posts bid/ask quotes on a multi-level order book and earns the spread on client fills.
- **Market B** — a reference market with a wide spread (75% vol weight, 200 ms refresh). Used for hedging large inventory positions.
- **Market C** — a fast reference market with a tight spread (25% vol weight, 170 ms refresh). Used for urgent hedges.
- **Client flow** — a Poisson-driven stream of market orders whose size and direction depend on the mid price, spread, and time of day.
- **HFT agents** *(Phase 3)* — high-frequency traders competing for fills on exchange A, following a realistic intraday activity schedule.

### Simulation phases

| Phase | Config | Stock path | Purpose |
|---|---|---|---|
| Phase 1 | Heuristic `QuoterConfig` | Seed | Baseline run with default parameters |
| Calibration | — | — | Fits MM parameters from Phase 1 data |
| Phase 1 Calibrated | Calibrated config | Same seed | Before/after comparison on identical conditions |
| Phase 2 | Calibrated config | Seed P2 | Out-of-sample test on a fresh price path |
| Phase 3 | Calibrated config + HFT | Seed P2 | Full competitive environment with HFT agents |

---

## 2. Repository Structure

```
fx_market_making/
│
├── main.py                    # Entry point for the desktop app
├── run_simulation.py          # Standalone terminal simulation script
├── README.md                  # This file
├── README_simulation.md       # Guide for run_simulation.py
│
├── app/                       # PySide6 desktop application
│   ├── main.py                # QApplication setup and dark theme
│   ├── main_window.py         # Main window: parameters + phase buttons
│   ├── workers.py             # QThread workers (one per phase)
│   ├── result_window.py       # Tabbed result window (report + figures)
│   ├── style.qss              # Dark VS Code-inspired stylesheet
│   └── README_app.md          # Full app user guide
│
├── src/utils/                 # Core simulation library
│   ├── stock_simulation/      # Price path generators (GBM, Heston, GARCH)
│   ├── market_simulator/      # Market B & C (spread, depth, mid price)
│   ├── order_book/            # Exchange A order book implementation
│   ├── market_maker/          # Quoter: quoting logic, fill handling, hedging
│   ├── client_flow/           # Poisson client order flow generator
│   ├── calibration/           # Calibrators: vol, spread, k, stale, gamma
│   ├── hft/                   # HFT agent, config, and intraday schedules
│   ├── report/                # Controller, P&L tracker, diagnostics, figures
│   └── simulation/            # Session persistence (save/load)
│
└── test/                      # Development and exploration notebooks
    ├── price_simulation.ipynb
    ├── order_book.ipynb
    ├── client_flow.ipynb
    ├── simulation.ipynb
    ├── phase2_calibration.ipynb  # ← interesting to explore calibration details
    └── phase3_hft.ipynb          # ← interesting to explore HFT scenario analysis
```

---

## 3. Installation

Python 3.12 is required. All dependencies are managed with `uv`.

```bash
# Install uv (if not already installed)
pip install uv

# Install all project dependencies
uv sync
```

The desktop app additionally requires PySide6:

```bash
pip install PySide6
```

---

## 4. How to Run

There are two ways to run the full simulation pipeline.

---

### Option A — Desktop App *(recommended)*

A full GUI with parameter controls, live progress bars, and interactive tabbed result windows.

```bash
python3.12 main.py
```

See **[app/README_app.md](app/README_app.md)** for the complete user guide.

---

### Option B — Standalone Script

A terminal script that runs the exact same pipeline sequentially. It pauses between phases so you can review results before continuing. Edit the parameters block at the top of the file before running.

```bash
python3.12 run_simulation.py
```

See **[README_simulation.md](README_simulation.md)** for details.

---

> **Note on the notebooks in `test/`:** The Jupyter notebooks were used during development to build and validate each component in isolation. They are not the primary way to run the project, but `test/phase2_calibration.ipynb` and `test/phase3_hft.ipynb` are worth exploring if you want to study the calibration pipeline or the HFT scenario analysis in detail.

---

## 5. Simulation Pipeline

### Step 1 — Price path simulation

A EUR/USD mid price is generated using one of three models:

- **GBM** — Geometric Brownian Motion. Log-normal diffusion with constant volatility.
- **Heston** — Stochastic volatility. Mean-reverting variance process; captures vol clustering without jumps.
- **GARCH** — GARCH(1,1) with optional jump component. Captures volatility clustering, fat tails, and sudden price shocks.

The path is discretised at a fixed time step `dt` (default: 0.05 s) over `N_DAYS` trading days.

---

### Step 2 — Market B & C construction

Two reference markets are built from the same price path:

- **Market B** — wider spread (Skew model, 75% vol weight), deeper book (500k EUR mean depth). Used for hedging when inventory is large.
- **Market C** — tighter spread (Adaptive model, 25% vol weight), shallower book (200k EUR mean depth). Used for urgent hedges.

Both markets expose a bid, ask, and multi-level depth at every time step.

---

### Step 3 — Phase 1 (heuristic quoting)

The MM runs on exchange A with a default `QuoterConfig`:

1. At each step, `compute_quotes()` selects which levels to reprice using a 5-priority system.
2. Stale quotes are cancelled; fresh bid/ask levels are posted.
3. Client orders arrive via `ClientFlowGenerator` (Poisson process, log-normal size).
4. Fills trigger `on_fill()`: inventory is updated, hedges are scheduled.
5. Hedges execute on Market B or C depending on urgency and inventory level.

P&L is tracked continuously: spread income, inventory mark-to-market, and exchange fees.

---

### Step 4 — Calibration

`CalibratedConfigBuilder` fits a new `QuoterConfig` from Phase 1 data. Five calibrators run in parallel:

| Calibrator | What it fits |
|---|---|
| `VolatilityCalibrator` | EWMA volatility → dynamic spread width |
| `SpreadCalibrator` | OLS regression on effective spreads → target half-spread |
| `KCalibrator` | Fill-rate intensity k → Poisson arrival rate model |
| `StaleCalibrator` | Stale-quote threshold → quote refresh frequency |
| `GammaOptimizer` | Inventory penalty γ → Avellaneda-Stoikov optimal skew |

---

### Step 5 — Phase 1 Calibrated (comparison)

The MM runs again on the **same stock path** with the calibrated config. Comparing Phase 1 and Phase 1 Calibrated on identical market conditions isolates the direct impact of calibration.

---

### Step 6 — Phase 2 (out-of-sample)

The calibrated config is tested on a **fresh stock path** (Seed P2). This is the true out-of-sample evaluation: the config was fitted on Seed 32 and is now tested on Seed 99.

---

### Step 7 — Phase 3 (HFT competition)

HFT agents are added to exchange A alongside the MM. They follow a realistic intraday schedule — active at open and close, quieter at lunch — and can operate in four states:

| HFT State | Behaviour |
|---|---|
| `ACTIVE` | Posting competitive quotes on both sides |
| `ONE_SIDED_BID` | Quoting bids only (inventory squeeze) |
| `ONE_SIDED_ASK` | Quoting asks only (inventory squeeze) |
| `OFFLINE` | Not quoting (vol spike or connectivity loss) |

This reduces the MM's fill rate and tests the robustness of the calibrated strategy under competitive pressure.

---

## 6. Module Reference

### `stock_simulation/`
Generates the EUR/USD mid price path. `Stock` is the main class; call `simulate_gbm()`, `simulate_heston()`, or `simulate_garch()` to produce the path array.

### `market_simulator/`
Builds Markets B and C from a `Stock` object. Each `Market` exposes bid/ask and depth arrays at every time step. Use `build_markets_B_C(stock)` from `report.fast_config` to build both in one call.

### `order_book/`
Implements exchange A's order book (`Order_book`). Handles MM quote posting, client order routing, fill detection, and order ageing.

### `market_maker/`
`Quoter` manages the full MM lifecycle: selective quote repricing (5-priority), fill handling, inventory tracking, hedge execution, and P&L accounting. `QuoterConfig` holds all tunable parameters.

### `client_flow/`
`ClientFlowGenerator` produces client market orders at each step via a Poisson arrival process. Order size follows a log-normal distribution calibrated to spread and time of day.

### `calibration/`
Five independent calibrators each fit one aspect of `QuoterConfig` from Phase 1 `Controller` data. `CalibratedConfigBuilder` runs them in parallel and assembles the final config.

### `hft/`
`HFTAgent` posts competitive quotes on exchange A following an intraday schedule. `HFTConfig` controls quote size and the volatility threshold above which the agent goes offline. `make_realistic_schedule(n_days)` builds a schedule that scales to any simulation length.

### `report/`
`Controller` is the simulation engine: wires together the order book, MM, client flow (and optionally HFT), and drives the time loop. Call `ctrl.report()` after `ctrl.simulate(n_steps)` to print the P&L breakdown and generate all diagnostic figures. `PnLTracker` augments the trade history dataframe with cumulative P&L columns.

### `simulation/`
`save_session` / `load_session` persist a full simulation result to disk for later analysis without re-running.

---

## 7. Parameters Reference

| Parameter | Description | Default |
|---|---|---|
| `MODEL` | Price process: `GARCH`, `GBM`, or `Heston` | `GARCH` |
| `SEED` | Stock path seed for Phase 1 and Comparison | `32` |
| `SEED_P2` | Stock path seed for Phase 2 and Phase 3 | `99` |
| `N_DAYS` | Simulation duration for Phase 1 & Comparison | `30` |
| `DT_SEC` | Time step for Phase 1 & Comparison (seconds) | `0.1` |
| `N_DAYS_P2` | Simulation duration for Phase 2 & 3 | `15` |
| `DT_SEC_P2` | Time step for Phase 2 & 3 — finer grid for HFT (seconds) | `0.05` |
| `VOL` | Annualised volatility | `0.07` |
| `ORIGIN` | Starting EUR/USD price | `1.10` |
| `CAPITAL` | Market-maker capital K in EUR/USD | `1 000 000` |
| `HFT_QUOTE_SIZE_EUR` | EUR notional per side per HFT agent (Phase 3) | `2 000` |

**GARCH parameters** (active when `MODEL = "GARCH"`):

| Parameter | Description | Default |
|---|---|---|
| `GARCH_ALPHA` | ARCH coefficient α | `0.05` |
| `GARCH_BETA` | GARCH coefficient β — requires α + β < 1 | `0.94` |
| `GARCH_LAM` | Jump intensity in jumps/year (0 = no jumps) | `100` |
| `GARCH_SIGMA_J` | Jump size standard deviation (log-return) | `0.005` |

**Heston parameters** (active when `MODEL = "Heston"`):

| Parameter | Description | Default |
|---|---|---|
| `HESTON_KAPPA` | Mean-reversion speed of variance | `2.0` |
| `HESTON_THETA` | Long-run variance (set to VOL² by default) | `0.0049` |
| `HESTON_XI` | Volatility of volatility | `0.3` |
| `HESTON_RHO` | Correlation between price and variance | `-0.1` |
