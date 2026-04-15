# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FX market making simulation (M2 Electronic Markets course). Models a market maker on Exchange A quoting EUR/USD using Avellaneda-Stoikov, hedging on exchanges B and C, with a full limit order book simulation underneath.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                                              # install dependencies
uv run main.py                                       # run main entry point
uv run jupyter notebook test/order_book.ipynb        # interactive order book notebook
uv add <package>                                     # add a dependency
```

No tests configured yet.

## Architecture

All simulation code lives in `src/utils/` with four sub-packages:

```
src/utils/
├── order_book/          # limit order book engine
├── stock_simulation/    # GBM price path
├── market_simulator/    # noised mid price + spread generation
└── market_maker/        # Avellaneda-Stoikov quoter
```

### `order_book/` — `order_book_impl.py`

Two classes: `Order` (value object) and `Order_book` (DataFrame-backed book).

**`Order` fields:** `id` (`{timestamp_ms}_{4-digit}` format), `direction`, `price`, `size`, `type`, `origin` (`"market_maker"` | `"client"`), `time`.

**`Order_book` key state:**
- `_df_order_book` — all live orders; `_df_bid_book` / `_df_ask_book` are filtered views
- `_df_matches` — filled trade history (`MatchId`, `ClientOrderId`, `MmOrderId`, `Direction`, `Price`, `MatchedSize`, `Time`)

**Generation / submission workflow:**
```python
ob._generate_n_random_order(30)                          # seed MM book (sequential, adds directly)
orders = ob.generate_random_orders(10, origin="client")  # build list from current book snapshot
ob.add_orders_batch(orders)                              # add list → triggers _try_clear()
```

**Matching rules** (`_try_clear`): client orders only match against MM orders, price-time priority, partial fills supported. Fill price = MM passive price.

**Cancellation:** `cancel_orders(ids)` by ID list; `cancel_all_mm_orders()` for full repricing.

**Price generation:** truncated normal anchored on last seen price of each side. MM bids capped at `best_ask - 0.0001`, MM asks floored at `best_bid + 0.0001` — book never crosses. Prices rounded to 4 decimal places.

---

### `stock_simulation/` — `stock.py`

`Stock` simulates a GBM price path.

| Attribute | Description |
|-----------|-------------|
| `self.simulation` | 1D array `(N+1,)` — the price path |
| `self._time_grid` | 1D array `(N+1,)` — time in seconds |
| `self.time_step` | `dt` in seconds |
| `self.n_steps` | `N` |
| `self.vol_realized` | per-step log-return residuals `(N,)` |

Key methods: `simulate_gbm(n_days, dt_seconds)`, `plot_path()`, `sanity_check()`.

**IMPORTANT:** the attribute is `self.simulation` (singular), not `self.simulations`.

---

### `market_simulator/` — `market.py` + `spread_utils.py`

`Market` wraps a `Stock` and builds observed bid/ask price arrays.

Spread variants stored as array attributes after calling the corresponding builder:
- `bid/ask_price_constant` — fixed tick spread
- `bid/ask_price_sto` — stochastic spread driven by rolling realized vol
- `bid/ask_price_adaptive` — adaptive vol window
- `bid/ask_price_asym` — asymmetric mean-reversion (widens fast, tightens slow)

`spread_utils.py` provides:
- `compute_rv_zero_mean(true_path, window_size, dt)` — zero-mean annualized rolling realized vol (preferred over `np.std` at high frequency)
- `evolve_s_excess(s_star, kappa_u, kappa_d, dt)` — sequential asymmetric mean-reversion for excess spread; `kappa_u >> kappa_d`

---

### `market_maker/` — `quoter.py`

`Quoter` implements Avellaneda-Stoikov with latency premium and inventory skew.

`QuoterConfig` (dataclass) holds all tunable parameters:

| Parameter | Default | Role |
|-----------|---------|------|
| `gamma` | 0.1 | risk-aversion |
| `k` | 1.5 | order arrival decay |
| `n_levels` | 10 | price levels each side |
| `beta` | 0.3 | size decay across levels |
| `Q_base` | 100 000 EUR | base size at best level |
| `tick_size` | 0.0001 | 1 bp |
| `delta_limit` | 0.90 | hedge trigger: `\|q\| / K` |
| `vol_window` | 6000 steps | EWMA realized vol window |

**`compute_quotes(step, t)`** — called once per simulation step; returns `(list[Quote], cancel_ids)`. Returns `([], [])` if best bid moved less than `requote_threshold` ticks (avoids unnecessary cancel-replace).

**`hedge_order(depth_B, depth_C, fair_mid)`** — splits hedge across B and C using score = `1 / (taker_fee + impact/depth)`; falls back to 75/25 volume weights if depth is zero.

**Vol estimation:** falls back to `stock.vol` for the first `vol_window` steps; switches to EWMA realized vol (computed on `market_B.noised_mid_price`) thereafter, floored at `0.2 × stock.vol`.

**Spread formula:**
```
reservation_price  = fair_mid - q·γ·σ²·(T-t)
spread_AS          = γ·(σ·100)²·(T-t) + (2/γ)·ln(1 + γ/k)   [in bps, then converted]
spread_latency     = 2·σ_per_s·√(effective_gap_s)
half_spread        = max((spread_AS + spread_latency) / 2, tick_size)
```

---

## Key Conventions

- Notebook working directory is `src/`, so all imports are relative: `import utils.order_book.order_book_impl as book_utils`
- Use `importlib.reload()` after editing modules to pick up changes without restarting the kernel
- All plots use a dark theme (`#111111` background)
- `TRADING_SECONDS_PER_YEAR` and `TRADING_SECONDS_PER_DAY` are in `stock_simulation/config.py`
