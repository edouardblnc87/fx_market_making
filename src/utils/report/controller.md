# Controller — method reference

## Simulation

| Method | What it does |
|---|---|
| `__init__(market_B, market_C, book, quoter, client_flow_fn)` | Wires all components together. Initialises internal counters (`_n_fills_prev`, `_n_quotes_posted`) and the empty step log. |
| `simulate()` | Main loop. Iterates `s` from 0 to `n_steps − 1`, calls `step(s, s * dt)` at each tick. |
| `step(step, t)` | One simulation tick: ticks the book, computes and posts MM quotes, reads best bid/ask, routes client orders, triggers hedge if needed, logs state. |

## Tracking what is happening

| Method / property | What it tracks |
|---|---|
| `_log_step(step, t, bid_A, ask_A)` | Appends one row per tick to `_step_log`. Captures: time, market A/B/C bid-ask-mid, fair mid, inventory, number of resting MM orders, fills this step, cumulative quotes posted. Called automatically at the end of every `step()`. |
| `step_log` *(property)* | Returns `_step_log` as a DataFrame — one row per simulation step. This is the main source for time-series plots. |
| `trade_history` *(property)* | Delegates to `quoter.trade_history`. Returns all fill events (MM fills + hedge legs) as a DataFrame with columns: `t`, `step`, `direction`, `price`, `size`, `level`, `is_hedge`, `is_full_fill`, `fair_mid`. |
| `pnl_report()` | Calls `PnLTracker.report()` on the trade history. Returns a dict with total MtM, realized, unrealized, inception spread, inventory revaluation, fees (maker A + taker B/C), fill counts. |

## Plots

| Method | What it shows |
|---|---|
| `plot_market_quotes()` | Three stacked panels (A, B, C) — bid, ask, mid over time. Market A shows resting MM quotes; B and C show simulated reference prices. |
| `plot_top_trades(n=10)` | Market A bid/ask/mid with the top `n` MM fills by size marked as arrows (buy = up green, sell = down red). |
| `plot_price_inventory()` | Fair mid price (left axis) vs EUR inventory (right axis) with ±delta limit lines. |
| `plot_mtm_percentiles()` | Per-trade MtM P&L distribution as a function of time since fill inception — mean, median, 5th/95th percentile band. |
| `fill_rate_analysis(plot=True)` | Fill count and relative fill rate by ladder level. Also returns a stats dict (`total_mm_fills`, `overall_fill_rate`, `avg_fill_size_eur`, `full_fill_pct`, `partial_fill_pct`). |

## Report

| Method | What it does |
|---|---|
| `report()` | Prints the full P&L summary table, then calls all five plot methods in sequence. Entry point for end-of-simulation analysis. |
