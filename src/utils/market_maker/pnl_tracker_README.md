# PnLTracker

Static analysis class for P&L decomposition of a market-making session.
All methods operate on `Quoter.trade_history` (a DataFrame). No state.

---

## P&L Decomposition

Two equivalent identities:

```
Total MtM P&L  =  realized_pnl  +  unrealized_pnl

               =  inception_spread_pnl  +  inventory_revaluation_pnl  −  total_fees
```

| Term | Definition |
|---|---|
| **Realized P&L** | Cash already locked in: `sum(cash_flow)` across all trades |
| **Unrealized P&L** | Open EUR inventory valued at current mid: `inventory × current_mid` (USD) |
| **MtM P&L** | `realized + unrealized` — true economic P&L if closed now |
| **Inception spread** | Spread captured at fill time: `(price − fair_mid) × size` for sells, `(fair_mid − price) × size` for buys — MM fills on A only |
| **Inventory revaluation** | P&L from fair_mid moving after fills: `MtM − inception + fees` |
| **Total fees** | All fees paid: maker on A + taker on B/C |

---

## Methods

| Method | Signature | Returns |
|---|---|---|
| `augment(df)` | `df → DataFrame` | Adds `cum_cash`, `cum_fees`, `mtm_pnl` columns (running per row) |
| `realized_pnl(df)` | `df → float` | `sum(cash_flow)` |
| `unrealized_pnl(df, current_mid)` | `df, float → float` | `final_inventory × current_mid` |
| `mtm_pnl(df, current_mid)` | `df, float → float` | `realized + unrealized` |
| `inception_spread_pnl(df)` | `df → float` | Spread captured at fill inception (MM fills only) |
| `inventory_revaluation_pnl(df, current_mid)` | `df, float → float` | Residual: `MtM − inception + fees` |
| `final_inventory_value(df, current_mid)` | `df, float → float` | `final_inventory × current_mid` (USD) |
| `report(df, current_mid)` | `df, float → dict` | Full breakdown (all above + fee split + trade counts) |

---

## Usage

```python
from utils.market_maker.pnl_tracker import PnLTracker

df = mm.trade_history                  # from Quoter after simulation
current_mid = df['fair_mid'].iloc[-1]  # or read from market at last step

# Add running MtM column to the history
aug = PnLTracker.augment(df)
print(aug[['t', 'cum_cash', 'mtm_pnl', 'inventory_after']].tail())

# Full end-of-session report
rep = PnLTracker.report(df, current_mid)
# {
#   'realized_pnl':              ...,   # cash locked in
#   'unrealized_pnl':            ...,   # open inventory value
#   'total_mtm_pnl':             ...,   # realized + unrealized
#   'inception_spread_pnl':      ...,   # spread at fill time
#   'inventory_revaluation_pnl': ...,   # price drift after fills
#   'total_fees':                ...,
#   'mm_maker_fees':             ...,
#   'hedge_taker_fees':          ...,
#   'final_inventory_eur':       ...,
#   'final_inventory_usd':       ...,
#   'n_mm_fills':                ...,
#   'n_hedges':                  ...,
#   'n_total_trades':            ...,
# }
```

---

## Notes

- `current_mid` should be the fair mid at the moment of evaluation (not at last fill). Use `mm.snapshot(step, t)['fair_mid']` or read directly from `market_B.bid_price[step]`.
- Hedge legs are excluded from inception spread (they trade at fair mid by construction, so their inception spread ≈ 0 before fees).
- The `augment` MtM column uses `fair_mid` at each fill as the mark price — this gives a per-trade MtM, not the final MtM at a single point in time.
