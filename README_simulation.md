# FX Market Making — Standalone Simulation Script

`run_simulation.py` is a terminal alternative to the desktop app (`app/`).  
It runs the exact same pipeline end-to-end, opens the same tabbed result windows after each phase, and pauses between steps so you can review results before continuing.

---

## How to run

```bash
python3.12 run_simulation.py
```

> PySide6 is only installed for Python 3.12 — do not use `python3` or `python`.

---

## How to configure

All parameters are at the top of `run_simulation.py` in a clearly delimited block.  
Edit them directly in the file before running — no UI needed.

```python
MODEL    = "GARCH"    # "GARCH" | "GBM" | "Heston"
SEED     = 32         # stock seed for Phase 1 and Comparison
SEED_P2  = 99         # stock seed for Phase 2 & 3 (out-of-sample)
N_DAYS    = 30        # trading days for Phase 1 & Calibrated Comparison
DT_SEC    = 0.1       # time step for Phase 1 & Calibrated Comparison (seconds)
N_DAYS_P2 = 15        # trading days for Phase 2 & 3 (shorter due to finer grid)
DT_SEC_P2 = 0.05      # time step for Phase 2 & 3 — finer resolution for HFT (seconds)
VOL      = 0.07       # annualised volatility
ORIGIN   = 1.10       # starting EUR/USD price
CAPITAL  = 1_000_000  # market-maker capital

# Toggle individual phases on/off
RUN_PHASE1            = True
RUN_CALIBRATION       = True
RUN_PHASE1_CALIBRATED = True
RUN_PHASE2            = True
RUN_PHASE3_HFT        = True
```

---

## Pipeline

---

### ▶ STEP 1 — Stock simulation

Simulates the EUR/USD price path (Seed) and builds Markets B & C.

---

### ▶ STEP 2 — Phase 1

Market-maker with the heuristic `QuoterConfig` on the Seed path.  
→ Opens a result window. **Pauses — review results, then click Continue.**

---

### ▶ STEP 3 — Calibration

Fits a `QuoterConfig` from Phase 1 data (volatility, spread targets, fill-rate k, stale threshold).  
→ Opens a Calibration Summary window. **Pauses.**

---

### ▶ STEP 4 — Phase 1 Calibrated (Comparison)

Same Seed path, calibrated config — direct before/after comparison on identical market conditions.  
→ Opens a result window. **Pauses.**

---

### ▶ STEP 5 — Phase 2

Calibrated config on a fresh stock path (Seed P2) — out-of-sample test.  
→ Opens a result window. **Pauses.**

---

### ▶ STEP 6 — Phase 3 (HFT)

Same as Phase 2 with HFT agents added (realistic intraday activity schedule).  
→ Opens a result window.

---

After all phases, the script prints `Done` and enters the Qt event loop — all windows stay open until you close them.

---

## Result windows

Identical to the app: each phase opens a tabbed window with a **Report** text tab, one **Figure** tab per plot, and the **QuoterConfig summary** as the last tab. Figures are scrollable and have the full matplotlib toolbar (zoom, pan, save).

---

## Difference vs the app

| | `run_simulation.py` | App (`python3.12 main.py`) |
|---|---|---|
| Parameters | Edit the file directly | UI spinners |
| Execution | Sequential, terminal output | Background threads, live progress bar |
| Pause between phases | Yes — click Continue dialog | No — each phase triggered manually |
| Result display | Same tabbed result windows | Same tabbed result windows |
