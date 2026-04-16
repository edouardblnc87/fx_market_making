# simulation.ipynb — Phase 1 End-to-End Simulation

End-to-end Phase 1 simulation of exchange A: GARCH price path → reference markets B/C → MM quoting → client flow → hedging → P&L.

---

## Structure

### Setup cells
- Simulate a 1-day GARCH stock path (`dt = 0.01s`, ~864,000 steps)
- Build **market B** with Skew spread, **market C** with Adaptive spread (deepcopy of same stock)
- Define helpers:
  - `new_sim()` — fresh `Order_book` + `Quoter` + wired fill callback
  - `post_initial_ladder(book, mm, step)` — tick + compute + cancel + post, prints ladder size
  - `show_history(mm)` — print trade history with key columns

---

## 4 Controlled Demo Cases

Each case calls `new_sim()` for a clean state.

### Case 1 — Full fill only
Manually craft a client buy order for the **entire** level-1 ask size.
Expected: `_pending_fills=1`, `_pending_topups=0`. On next step, the filled slot is re-quoted (or the full ladder is repriced if price drifted past threshold).

### Case 2 — Partial fill only
Manually craft a client sell order for **half** of the level-1 bid size.
Expected: `_pending_fills=0`, `_pending_topups=1`. On next step, a top-up order is posted at the same price for the gap size (`original_size − remaining_size`).

### Case 3 — Full fill + partial fill in the same step
Both a full fill (ask L1) and a partial fill (bid L1) arrive at step 0.
Both are handled in one Priority-4 pass: re-quote + top-up emitted together.

### Case 4 — Forced hedge
Pre-load `mm.inventory = CAPITAL * 0.91` (above 90% limit), call `mm.execute_hedge()`.
Expected: hedge legs appear in trade history with `is_hedge=True`, `venue="B"/"C"`. Inventory returns to 0 (flat / 50-50).

---

## Full Day Simulation Loop

```
for each step:
    book.tick(step)                          # age resting orders
    quotes, cancels = mm.compute_quotes(…)   # selective requote (priority 1-5)
    book.cancel_orders(cancels)
    book.post_mm_quotes(quotes)
    for order in gen.generate_step(…):       # Poisson client flow
        book.route_client_order(order)       # add + try_clear → may fire on_fill
    mm.execute_hedge(step, t, mid)           # hedge if |inv| > 90%
```

`on_fill` fires synchronously inside `route_client_order` → inventory and `_fill_history` are updated before the next step's `compute_quotes`.

---

## P&L Chart (3 panels)

After the full-day run:
1. **Cumulative cash P&L** — `cash_flow.cumsum()` with hedge dots marked in red
2. **Inventory (EUR)** — `inventory_after` over time with ±90% limit lines
3. **Cumulative fees** — `fee_cost.cumsum()`

---

## Using PnLTracker

```python
from utils.market_maker.pnl_tracker import PnLTracker

df = mm2.trade_history
current_mid = df['fair_mid'].iloc[-1]

aug = PnLTracker.augment(df)        # adds cum_cash, cum_fees, mtm_pnl columns
rep = PnLTracker.report(df, current_mid)

print(f"Realized P&L : {rep['realized_pnl']:.2f} USD")
print(f"MtM P&L      : {rep['total_mtm_pnl']:.2f} USD")
print(f"Inception     : {rep['inception_spread_pnl']:.2f} USD")
print(f"Revaluation  : {rep['inventory_revaluation_pnl']:.2f} USD")
print(f"Total fees   : {rep['total_fees']:.2f} USD")
```

See `pnl_tracker_README.md` for full method reference.
