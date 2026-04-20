# FX Market Making — Desktop Application

A PySide6 dark-theme desktop app that runs the full simulation pipeline interactively, with live progress bars and tabbed result windows for each phase.

---

## How to run

```bash
python3.12 main.py
```

> PySide6 is only installed for Python 3.12 — do not use `python3` or `python`.

---

## Parameters

Set all parameters before running. They are read at the moment each button is clicked.

| Field | Description | Default |
|---|---|---|
| Model | Price process: `GARCH`, `GBM`, or `Heston` | GARCH |
| Seed | Stock path seed — used by Phase 1 and the Comparison step | 32 |
| Seed P2 | Fresh stock seed — used by Phase 2 and Phase 3 | 99 |
| N Days | Simulation duration for Phase 1 & Comparison (trading days) | 30 |
| DT (s) | Time step for Phase 1 & Comparison (seconds) | 0.1 |
| N Days P2 | Simulation duration for Phase 2 & 3 (trading days) | 15 |
| DT P2 (s) | Time step for Phase 2 & 3 — finer grid for HFT (seconds) | 0.05 |
| Vol | Annualised volatility | 0.07 |
| Origin | Starting EUR/USD price | 1.10 |
| Capital K (EUR/USD) | Market-maker initial capital | 1 000 000 |
| Quote size / side (EUR) | EUR notional per side posted by each HFT agent (Phase 3 only) | 2 000 |

GARCH fields (α, β, λ, σJ) and Heston fields (κ, θ, ξ, ρ) appear automatically when the corresponding model is selected.

---

## Workflow

Follow the steps in order. Each button is greyed out until its prerequisites are met.

---

### ▶ STEP 1 — Simulate Stock

Simulates the EUR/USD price path using the chosen model and Seed.

---

### ▶ STEP 2 — Build Markets

Constructs Markets B and C (spread arrays, depth levels) from the simulated stock.  
Required before running any phase.

---

### ▶ STEP 3 — Run Phase 1

Runs the market-maker with the **heuristic** (default) `QuoterConfig` on the stock path from Seed.

Opens a result window with:
- **Report tab** — full backtesting report (P&L, spreads, fill rates)
- **Figure tabs** — all diagnostic plots
- **Last tab** — QuoterConfig parameter summary

---

### ▶ STEP 4 — Run Calibration

Fits a `QuoterConfig` from Phase 1 data:
volatility EWMA, spread targets, fill-rate intensity k, stale-quote threshold.

Opens a **Calibration Summary** window showing all calibrated parameter values.

*Requires Phase 1.*

---

### ▶ STEP 5 — Phase 1 Calibrated — Compare

Re-runs the market-maker on the **exact same stock path** (Seed) with the **calibrated** config.  
Direct before/after comparison: identical market conditions, updated quoting parameters.

Opens a result window labelled "Comparison".

*Requires Calibration.*

---

### ▶ STEP 6 — Run Phase 2

Runs the market-maker with the calibrated config on a **fresh stock path** (Seed P2).  
Out-of-sample test: config calibrated on Seed 32, tested on Seed 99.

Falls back to the default `QuoterConfig` if calibration has not been run.

Opens a result window.

---

### ▶ STEP 7 — Run Phase 3 (HFT)

Same as Phase 2 but adds HFT agents with a realistic intraday activity schedule.  
HFT agents post quotes on both sides at the configured "Quote size / side".

Opens a result window with HFT-specific figures (agent activity, fill competition, state timeline).

---

## Result windows

Each phase opens an **independent** tabbed window:

- **Report** — full backtesting text, scrollable, monospace font
- **Figure 1 … N** — each figure in a scrollable canvas with the matplotlib toolbar (zoom, pan, save)
- **Last figure** — QuoterConfig parameter summary

Multiple windows can be open at the same time and closed independently.

---

## Status bar

| Indicator | Meaning |
|---|---|
| Pulsing bar | Indeterminate progress (stock simulation, market build) |
| 0 – 100 % bar | Determinate progress (phase simulations, driven by tqdm steps) |
| Text | Current operation and elapsed seconds |
