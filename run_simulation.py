"""
run_simulation.py — Standalone FX Market Making simulation script.

Runs the full pipeline in sequence:
    1. Stock simulation
    2. Market B & C construction
    3. Phase 1  — heuristic QuoterConfig
    4. Calibration — fit QuoterConfig from Phase 1 data
    5. Phase 2  — calibrated QuoterConfig on a fresh stock path
    6. Phase 3  — calibrated MM + HFT agents (realistic schedule)

All parameters are at the top of this file.
Run with:
    python3.12 run_simulation.py
"""

import sys
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

# ===========================================================================
#  PARAMETERS — edit everything in this block
# ===========================================================================

# ── Stock & market ──────────────────────────────────────────────────────────
MODEL    = "GARCH"   # "GARCH" | "GBM" | "Heston"
SEED     = 32        # stock seed (also used as client-flow seed for Phase 1)
SEED_P2  = 99        # stock seed for Phase 2 & 3 (different path → fair test)
N_DAYS   = 30        # simulation duration in trading days
DT_SEC   = 0.1       # time step in seconds
VOL      = 0.07      # annualised volatility (EUR/USD baseline ≈ 7%)
ORIGIN   = 1.10      # starting EUR/USD price
CAPITAL  = 1_000_000 # market-maker capital K in EUR/USD

# ── GARCH + jump parameters (used only when MODEL = "GARCH") ───────────────
GARCH_ALPHA   = 0.05    # GARCH(1,1) α
GARCH_BETA    = 0.94    # GARCH(1,1) β  (α+β < 1 → stationary)
GARCH_LAM     = 100.0   # jump intensity in jumps/year  (0 = no jumps)
GARCH_SIGMA_J = 0.005   # jump size std (log-return)

# ── Heston parameters (used only when MODEL = "Heston") ────────────────────
HESTON_KAPPA = 2.0      # mean-reversion speed
HESTON_THETA = VOL**2   # long-run variance (= vol² → same baseline)
HESTON_XI    = 0.3      # vol-of-vol
HESTON_RHO   = -0.1     # price/vol correlation

# ── Phase 3 — HFT ──────────────────────────────────────────────────────────
HFT_QUOTE_SIZE_EUR = 2_000   # EUR notional per side posted by each HFT

# ── What to run ─────────────────────────────────────────────────────────────
RUN_PHASE1       = True
RUN_CALIBRATION  = True   # requires RUN_PHASE1 = True
RUN_PHASE2       = True
RUN_PHASE3_HFT   = True

# ===========================================================================
#  END OF PARAMETERS
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def _build_stock(seed: int, n_days: int) -> "Stock":
    from utils.stock_simulation.stock import Stock
    np.random.seed(seed)
    stock = Stock(drift=0.0, vol=VOL, origin=ORIGIN)
    if MODEL == "GBM":
        stock.simulate_gbm(n_days=n_days, dt_seconds=DT_SEC)
    elif MODEL == "Heston":
        stock.simulate_heston(
            n_days=n_days, dt_seconds=DT_SEC,
            kappa=HESTON_KAPPA, theta=HESTON_THETA,
            xi=HESTON_XI, rho=HESTON_RHO,
        )
    else:
        stock.simulate_garch(
            n_days=n_days, dt_seconds=DT_SEC,
            alpha=GARCH_ALPHA, beta=GARCH_BETA,
            lam=GARCH_LAM, sigma_J=GARCH_SIGMA_J,
        )
    return stock


def _build_markets(stock):
    from utils.report.fast_config import build_markets_B_C
    return build_markets_B_C(stock)


def _run_phase(market_B, market_C, quoter_cfg, seed: int,
               hft: bool = False, n_days: float = N_DAYS):
    from utils.order_book.order_book_impl import Order_book
    from utils.market_maker.quoter import Quoter
    from utils.report.controller import Controller
    from utils.client_flow.flow_generator import ClientFlowGenerator
    from utils.hft.hft_config import HFTConfig
    from utils.hft.scenarios import make_realistic_schedule

    book = Order_book()
    mm   = Quoter(market_B, market_C, config=quoter_cfg, capital_K=CAPITAL)
    book.register_quoter_listener(mm.on_fill)
    gen  = ClientFlowGenerator(seed=seed)

    hft_kw = {}
    if hft:
        hft_kw = dict(
            hft=True,
            hft_config=HFTConfig(max_depth_eur=HFT_QUOTE_SIZE_EUR),
            hft_schedule=make_realistic_schedule(n_days),
        )

    ctrl = Controller(
        market_B, market_C, book, mm,
        lambda step, t, mid, bid, ask, dt:
            gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
        **hft_kw,
    )
    t0 = time.time()
    ctrl.simulate(market_B.stock.n_steps)
    print(f"  Simulation done in {time.time() - t0:.1f} s  "
          f"({market_B.stock.n_steps:,} steps)")
    return ctrl


# ---------------------------------------------------------------------------
# Step 1 — Stock simulation
# ---------------------------------------------------------------------------

_header(f"Step 1 — Stock simulation  [{MODEL}]")
print(f"  seed={SEED}  n_days={N_DAYS}  dt={DT_SEC}s  "
      f"vol={VOL}  origin={ORIGIN}")

stock_p1 = _build_stock(SEED, N_DAYS)
print(f"  {stock_p1.n_steps:,} steps generated")

market_B_p1, market_C_p1 = _build_markets(stock_p1)
print("  Markets B & C built")

# ---------------------------------------------------------------------------
# Step 2 — Phase 1 (heuristic parameters)
# ---------------------------------------------------------------------------

ctrl_p1 = None
if RUN_PHASE1:
    _header("Step 2 — Phase 1  (heuristic QuoterConfig)")
    from utils.market_maker.quoter import QuoterConfig
    ctrl_p1 = _run_phase(market_B_p1, market_C_p1, QuoterConfig(), seed=SEED)
    ctrl_p1.report()
    plt.show()

# ---------------------------------------------------------------------------
# Step 3 — Calibration
# ---------------------------------------------------------------------------

cal_cfg = None
if RUN_CALIBRATION:
    if ctrl_p1 is None:
        print("\n[SKIP] Calibration requires Phase 1 — set RUN_PHASE1 = True")
    else:
        _header("Step 3 — Calibration  (Phase 1 → QuoterConfig)")
        from utils.calibration.calibrated_config import CalibratedConfigBuilder
        builder = CalibratedConfigBuilder(ctrl_p1)
        cal_cfg = builder.build()
        print(builder.summary())

# ---------------------------------------------------------------------------
# Step 4 — Phase 2 (calibrated parameters, fresh stock path)
# ---------------------------------------------------------------------------

ctrl_p2 = None
if RUN_PHASE2:
    _header(f"Step 4 — Phase 2  (seed={SEED_P2}, "
            f"{'calibrated' if cal_cfg else 'default'} config)")

    from utils.market_maker.quoter import QuoterConfig
    cfg_p2 = cal_cfg if cal_cfg is not None else QuoterConfig()

    stock_p2 = _build_stock(SEED_P2, N_DAYS)
    market_B_p2, market_C_p2 = _build_markets(stock_p2)
    print(f"  Fresh stock: {stock_p2.n_steps:,} steps  (seed={SEED_P2})")

    ctrl_p2 = _run_phase(market_B_p2, market_C_p2, cfg_p2, seed=SEED_P2)
    ctrl_p2.report()
    plt.show()

# ---------------------------------------------------------------------------
# Step 5 — Phase 3 (calibrated MM + HFT agents)
# ---------------------------------------------------------------------------

if RUN_PHASE3_HFT:
    _header(f"Step 5 — Phase 3  (HFT, quote size={HFT_QUOTE_SIZE_EUR} EUR/side)")

    from utils.market_maker.quoter import QuoterConfig
    cfg_p3 = cal_cfg if cal_cfg is not None else QuoterConfig()

    # Reuse Phase 2 markets if available, otherwise build fresh
    if ctrl_p2 is not None:
        market_B_p3, market_C_p3 = market_B_p2, market_C_p2
        seed_p3 = SEED_P2
        print(f"  Reusing Phase 2 markets (seed={SEED_P2})")
    else:
        stock_p3 = _build_stock(SEED_P2, N_DAYS)
        market_B_p3, market_C_p3 = _build_markets(stock_p3)
        seed_p3 = SEED_P2
        print(f"  Fresh stock built (seed={SEED_P2})")

    ctrl_p3 = _run_phase(
        market_B_p3, market_C_p3, cfg_p3, seed=seed_p3,
        hft=True, n_days=float(N_DAYS),
    )
    ctrl_p3.report()
    plt.show()

_header("Done")
