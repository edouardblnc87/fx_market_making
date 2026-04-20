# report — Simulation Orchestration, P&L, Diagnostics

## Context

All other modules (`stock_simulation`, `market_simulator`, `order_book`, `market_maker`, `client_flow`, `hft`) produce components. This module **runs them together**, records the result, and turns that result into P&L numbers, diagnostic plots, and parameter scorecards. It is the canonical entry point for every backtest — no notebook should contain a bare step-loop.

Three public classes, all exported from `__init__.py`:

| Class | Responsibility |
|---|---|
| [`Controller`](controller.py) | Orchestrates the simulation loop and records the step log. |
| [`PnLTracker`](pnl_tracker.py) | Static P&L decomposition on a `trade_history` DataFrame. |
| [`DiagnosticsReport`](diagnostics.py) | Visual scorecard — what happened, is the spread right, should parameters change. |

One helper module, [`fast_config.py`](fast_config.py), provides pre-baked factories (standard demo spreads, demo `QuoterConfig`, 1M capital) so notebooks can spin up a valid `Controller` in four lines.

---

## Architecture

```
                ┌──────────────── Controller ────────────────┐
                │                                            │
  market_B  ───▶│   step(s, t):                              │
  market_C  ───▶│     book.tick(s)                           │
  book      ───▶│     (hft.step)        [Phase 3 only]       │
  quoter    ───▶│     quoter.compute_quotes → post/cancel    │
  client_fn ───▶│     client_flow_fn  → route_client_order   │
  (hft opt) ───▶│     quoter.execute_hedge                   │
                │     _log_step                              │
                │                                            │
                │   simulate():  loop step() over n_steps    │
                │   step_log, trade_history, pnl_report      │
                └──────────────────────┬─────────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
          PnLTracker          DiagnosticsReport          ctrl.report()
        (numbers only)     (visual scorecard, plots)    (combines both)
```

`Controller.step()` is fixed in order:

1. `book.tick(step)` — age resting orders.
2. `hft_agent.step(step, t)` — Phase 3 only; HFT reprices first (1-step latency vs. our 2–4 steps).
3. `quoter.compute_quotes` → `book.cancel_orders` / `book.post_mm_quotes`.
4. Best bid/ask read from cache (`quoter._last_best_bid_A` / `_last_best_ask_A`) — no dict scan.
5. `client_flow_fn(step, t, mid_A, best_bid_A, best_ask_A, dt)` → `book.route_client_order` (fires `FillEvent`s synchronously).
6. `quoter.execute_hedge(step, t)` — unconditional call; hedge decides internally whether to fire.
7. `_log_step` — one row written to pre-allocated numpy arrays (downsampled to ≤5000 points for long runs).

**Do not reorder.** The quoter caches fair-mid during `compute_quotes`; `execute_hedge` reads that cache. The client flow must run after new quotes are posted or fills will match against stale levels.

---

## Controller — Public API

### Setup & Run

| Method | Purpose |
|---|---|
| `__init__(market_B, market_C, book, quoter, client_flow_fn, hft=False, hft_config=None, hft_schedule=None)` | Wires components together. Caches `dt`, `weight_B`, `weight_C` for the hot path. If `hft=True`, creates an `HFTAgent` and registers it with the book. |
| `step(step, t)` | One simulation tick — see the 7-step order above. |
| `simulate(limit=None)` | Loop `step()` from 0 to `n_steps − 1` (or `limit`). Pre-allocates log arrays, shows a tqdm progress bar. Log rows are capped at `_LOG_MAX_PTS = 5000` via `_log_stride` — fills are always accumulated, only the time-series snapshot is downsampled. |

### Introspection

| Attribute | Contents |
|---|---|
| `step_log` *(property)* | DataFrame, one row per logged step. Columns: `step`, `t`, `bid_A/ask_A/mid_A`, `bid_B/ask_B/mid_B`, `bid_C/ask_C/mid_C`, `fair_mid`, `inventory`, `n_mm_orders`, `fills_this_step`, `quotes_posted`, `hedge_fired`. **Primary debugging surface.** |
| `trade_history` *(property)* | Delegates to `quoter.trade_history`. One row per fill: `t`, `step`, `direction`, `price`, `size`, `level`, `is_hedge`, `is_full_fill`, `fair_mid`, `cash_flow`, `fee_cost`, `inventory_after`. |
| `pnl_report()` | Calls `PnLTracker.report(trade_history, current_mid)` — returns the full dict described below. |

### Plots (all rendered inside `report()`)

| Method | Shows |
|---|---|
| `plot_market_quotes()` | Three stacked panels (A, B, C): bid, ask, mid over time. |
| `plot_top_trades(n=10)` | Market A price context as faint background; top `n` fills by size as centre-stage markers (120–620 pt², white edge, drop-lines to mid, per-fill rank / size / bps annotations). |
| `plot_price_inventory()` | Fair mid (left axis) vs EUR inventory (right axis) with ±`delta_limit × K/2` hedge lines. |
| `plot_mtm_percentiles()` | Per-trade MtM evolution percentiles, computed on the `step_log` grid (`cash + Δinventory × mid` forward from fill time) — no n×n matrix. |
| `fill_rate_analysis(plot=False)` | Returns `{total_mm_fills, overall_fill_rate, avg_fill_size_eur, full_fill_pct, partial_fill_pct}` and prints a per-level text table. |
| `plot_config_summary()` | Styled parameter table (quoter, market, capital) rendered before the run so the PDF shows exactly which config produced the numbers. |

### Entry Point

| Method | Purpose |
|---|---|
| `report(phase=None)` | Prints `plot_config_summary`, the full P&L text table, fill-rate analysis text, then calls each plot method in sequence. Pass `phase="phase3"` (or a phase number) to branch into `_report_hft` and the Phase 3 HFT-specific plots. |
| `run_phase3(isolated_days=3)` | Convenience wrapper: runs the isolated-HFT scenario, then the realistic month, producing the Phase 3 comparison plots. |

---

## PnLTracker — Decomposition

`PnLTracker` is **stateless** — all methods are static and operate on a `trade_history` DataFrame. Two equivalent identities:

```
Total MtM P&L = realized + unrealized
              = inception_spread + inventory_revaluation − total_fees
```

| Term | Definition |
|---|---|
| **Realized** | `sum(cash_flow)` — cash already locked in (positive = cash in, negative = cash out, USD). |
| **Unrealized** | `final_inventory_eur × current_mid` — open EUR leg valued at current fair mid. |
| **Inception spread** | `(price − fair_mid) × size` for sells, `(fair_mid − price) × size` for buys — **MM fills on A only** (hedges trade at fair mid by construction). |
| **Inventory revaluation** | Residual `MtM − inception + fees` — P&L from fair_mid drift after fills. |
| **Total fees** | `mm_maker_fees` (A) + `hedge_taker_fees` (B/C). Fees are already baked into `cash_flow`; the separate `fee_cost` column is for reporting only — don't subtract twice. |

Full method list in [pnl_tracker_README.md](pnl_tracker_README.md). Most notable:

- `report(df, current_mid) → dict` — canonical end-of-session summary (the dict printed by `Controller.report()`).
- `plot(df, current_mid, step_log=ctrl.step_log)` — 4-panel session chart. **Always pass `step_log`** when available: it enables a continuous MtM curve via `_continuous_mtm`, computed at every logged step (far more informative than the sparse fill-event MtM).
- `per_trade_mtm_evolution(df)` — `n × n` matrix of per-fill contributions over time. Used by `plot_per_trade_mtm(top_n)`; heavy, call explicitly.

---

## DiagnosticsReport — Visual Scorecard

Constructed from a completed `Controller`: `DiagnosticsReport(ctrl).report()`. Four sections, four questions:

| Plot | Question answered |
|---|---|
| `plot_overview()` | What happened? Price path, fills, hedges, inventory on one timeline. |
| `plot_spread_and_skew()` | Is the spread right? MM spread vs. B/C reference, skew decomposition, premium over reference. |
| `plot_fills()` | Is the order flow as expected? Fill rate by ladder level, inter-arrival distribution, level utilisation, full/partial split. |
| `plot_hedges()` | When did we hedge, how big, at what cost? Inventory trajectory annotated with hedge events. |
| `plot_param_scorecard()` | Should I change any parameters? Colour-coded table of heuristics (fill rate, spread premium, inventory excursion, hedge frequency) with ranges and recommendations. |

Style is shared dark-theme rcParams (see top of [diagnostics.py](diagnostics.py)). Plots downsample to `_MAX_PTS = 3000` for responsiveness.

---

## fast_config — Notebook Bootstrap

[`fast_config.py`](fast_config.py) pre-bakes the standard demo setup so notebooks don't re-specify the same 20 parameters each time:

| Function | Returns |
|---|---|
| `build_markets_B_C(stock)` | `(market_B, market_C)`. B uses **Skewed** spread (`spread_bps=0.5`, `alpha=1.5`, `gamma=0.3`, `ema_span=10s`), C uses **Adaptive** (`spread_bps=0.7`, `alpha=0.8`). Built concurrently via `ThreadPoolExecutor`. Depth: 500k EUR on B, 200k EUR on C. |
| `get_standard_demo_config()` | `QuoterConfig(gamma=0.01, requote_threshold_spread_fraction=0.25, delta_limit=0.60, hedge_partial_limit=0.05, emergency_penalty_multiplier=5.0)`. |
| `build_market_maker_and_order_book(market_B, market_C)` | `(quoter, book)` with `capital_K = 1_000_000`, listener already registered. |
| `build_controller(market_B, market_C, book, quoter, seed=44)` | Returns a ready-to-run `Controller` with a picklable `_ClientFlowFn` (so `simulation.save_session` can pickle the full run). |

The demo config is deliberately **not a default** on `QuoterConfig` — these values belong to the demo scenario, not to the production quoter. Override by constructing your own `QuoterConfig` and passing it to `Quoter` directly.

---

## Key Invariants

- **`Controller` is the canonical entry point.** No notebook should hand-roll a step loop — the log schema, hedge ordering, and HFT integration are only correct through `Controller`.
- **`step_log` is the primary debug surface.** Inventory, fill counts, and quote counts per step live here. Fills themselves live in `trade_history` (one row per match).
- **Cash-flow sign convention is load-bearing.** Positive = cash in, negative = cash out, USD. Breaking this silently corrupts `realized_pnl`. See the project-level `CLAUDE.md` for the full convention.
- **Pass `step_log` to `PnLTracker.plot`.** Without it, the MtM curve is sparse (one point per fill); with it, MtM is interpolated at every logged step via `_continuous_mtm`.
- **Log is downsampled.** Long runs write every `_log_stride`-th step. The `fills_this_step` / `quotes_posted` counters are accumulated across skipped steps so no fill is lost — only the price/inventory snapshot is subsampled.

---

## Module Map

| File | Responsibility |
|---|---|
| [controller.py](controller.py) | `Controller` — simulation loop, step log, plot methods, `report()` entry point. |
| [pnl_tracker.py](pnl_tracker.py) | `PnLTracker` — static P&L decomposition and session plots. |
| [diagnostics.py](diagnostics.py) | `DiagnosticsReport` — visual scorecard (overview, spread, fills, hedges, parameters). |
| [fast_config.py](fast_config.py) | Demo factories (markets, config, controller) for fast notebook setup. |
| [controller.md](controller.md) | Method-level reference for `Controller` (kept as deep-dive companion). |
| [pnl_tracker_README.md](pnl_tracker_README.md) | Method-level reference and formulas for `PnLTracker` (kept as deep-dive companion). |

---

## Usage

```python
from utils.report import Controller, PnLTracker, DiagnosticsReport
from utils.report.fast_config import (
    build_markets_B_C, build_market_maker_and_order_book, build_controller
)

market_B, market_C = build_markets_B_C(stock)
quoter, book       = build_market_maker_and_order_book(market_B, market_C)
ctrl               = build_controller(market_B, market_C, book, quoter)

ctrl.simulate()
ctrl.report()                       # full report + plots
DiagnosticsReport(ctrl).report()    # visual scorecard on top
```

For method-level detail, read [controller.md](controller.md) (Controller) and [pnl_tracker_README.md](pnl_tracker_README.md) (PnLTracker).
