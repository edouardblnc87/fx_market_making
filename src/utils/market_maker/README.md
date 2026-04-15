# market_maker — Quoting Framework

## Context

Exchange A is a new venue that has approached our firm to act as Designated Market Maker. In Phase 1, the exchange has zero trading history — all quoting logic must be purely heuristic. We are the sole market maker on Exchange A, with two reference markets:

- **Exchange B** — 75% of total EUR/USD volume, data arrives with 200ms latency
- **Exchange C** — 25% of total EUR/USD volume, data arrives with 170ms latency

HFTs access both feeds in 50ms, giving them a 150ms/120ms edge over us.

The `Quoter` is designed to work across all three phases — it degrades gracefully when data is sparse (Phase 1) and improves automatically as history accumulates (Phase 2+).

---

## Quoting Framework — Step by Step

### Step 1 — Fair mid price

We compute the best bid and best ask across both venues (with latency offsets applied) and take their midpoint:

```
best_bid = max(bid_B(t − 200ms), bid_C(t − 170ms))
best_ask = min(ask_B(t − 200ms), ask_C(t − 170ms))
fair_mid = (best_bid + best_ask) / 2
```

### Step 2 — Adaptive volatility (EWMA)

Instead of a fixed parametric σ, the quoter estimates realized volatility using an Exponentially Weighted Moving Average (EWMA) of log-returns from the observed B feed:

```
log_rets = diff(log(noised_mid_price[step - vol_window : step]))
ewma_vol = ewm(span=vol_window).std() / sqrt(dt / T_year)
```

Before `vol_window` steps have elapsed, it falls back to the parametric `stock.vol`. A safety floor at 20% of parametric vol prevents degenerate near-zero spreads.

EWMA is preferred over a simple rolling std because it weights recent observations more heavily, making the estimate more responsive to sudden changes while remaining stable during quiet periods.

**Note on microstructure noise:** realized vol computed from observed prices is systematically higher than the true underlying vol due to bid-ask bounce — each observed price oscillates around the true mid, adding noise variance on top of the true variance. This is intentional. A real market maker observes noisy prices and should price that risk accordingly. Methods like TSRV (Two-Scale Realized Volatility) can correct for this bias but are beyond the scope of Phase 1.

### Step 3 — Reservation price (Avellaneda-Stoikov)

The reservation price is our inventory-adjusted fair value:

```
r(t) = fair_mid − q × γ × σ² × (T − t)
```

- `q` = EUR inventory (positive = long, negative = short)
- `γ` = risk-aversion coefficient
- `T − t` = time remaining in the session (in years)

If long EUR, r < fair_mid → we price cheaper to attract sellers. If short, r > fair_mid → we price higher to attract buyers.

### Step 4 — Total spread (two components)

**Component A — Avellaneda-Stoikov spread**

```
spread_AS = [γ × (σ×100)² × (T−t) + (2/γ) × ln(1 + γ/k)] / 10000 × fair_mid
```

Expressed in bps then converted to price units. The first term grows with variance × time remaining. The second term is a liquidity premium driven by `k` (order arrival decay rate).

**Component B — Latency premium**

We are on average 142.5ms behind HFTs (volume-weighted: 0.75×150ms + 0.25×120ms). We charge for this adverse selection risk:

```
effective_gap = 0.75 × 150ms + 0.25 × 120ms = 142.5ms
spread_latency = 2 × (σ / sqrt(T_year)) × sqrt(effective_gap)
total_spread = spread_AS + spread_latency
```

Since σ here is the EWMA realized vol, the latency premium also adapts to current market conditions — it widens automatically during volatile periods.

### Step 5 — 10 price levels with inventory-skewed sizing

Prices step away from best by one tick per level. Sizes are skewed based on current inventory to accelerate mean-reversion:

```
inventory_skew = clip(q / K, -1, 1)
bid_size_i = Q_base × exp(-β×i) × (1 − 0.5 × skew)   # shrink when long
ask_size_i = Q_base × exp(-β×i) × (1 + 0.5 × skew)   # grow when long
```

When flat (skew=0) the book is perfectly symmetric. When long, ask gets more size to attract buyers and reduce inventory. When short, bid gets more size to attract sellers.

### Step 6 — Requote threshold

To avoid unnecessary cancel/replace cycles when the price is stable, we only requote if the best bid has moved by more than `requote_threshold` ticks since the last quote:

```
if |best_bid - prev_best_bid| < requote_threshold × tick: do nothing
```

### Step 7 — Dynamic hedge routing (fee + depth aware)

When `|q| / K > 90%`, we hedge by sending market orders on B and/or C. The split is determined jointly by current depth and fees:

```
score_venue = 1 / (taker_fee + impact_factor / depth)
ratio_B = score_B / (score_B + score_C)
```

If B has deep liquidity and lower fees it gets most of the order. If B is thin at this moment, C gets more even though it's slightly more expensive in fees. If B is offline (depth=0), 100% routes to C. The fee cost of the hedge is returned explicitly so the simulator can deduct it from realized P&L.

---

## Classes

### `QuoterConfig`

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.1 | Risk-aversion coefficient |
| `k` | 1.5 | Order arrival decay rate |
| `T` | `TRADING_SECONDS_PER_DAY` | Session horizon |
| `n_levels` | 10 | Price levels each side |
| `beta` | 0.3 | Size decay across levels |
| `Q_base` | 100,000 | Base EUR size at best level |
| `tick_size` | 0.0001 | 1bp tick |
| `requote_threshold` | 2 | Min tick move to trigger requote |
| `vol_window` | 200 | EWMA span in steps for realized vol |
| `weight_B/C` | 0.75/0.25 | Volume weights |
| `latency_B/C_s` | 0.200/0.170 | Data latency in seconds |
| `latency_hft_s` | 0.050 | HFT latency |
| `delta_limit` | 0.90 | Hedge trigger threshold |
| `fee_A_maker` | 0.0001 | Exchange A maker fee (0.01%) |
| `fee_A_taker` | 0.0004 | Exchange A taker fee (0.04%) |
| `fee_B_maker` | 0.00009 | Exchange B maker fee (0.009%) |
| `fee_B_taker` | 0.0002 | Exchange B taker fee (0.02%) |
| `fee_C_maker` | 0.00009 | Exchange C maker fee (0.009%) |
| `fee_C_taker` | 0.0003 | Exchange C taker fee (0.03%) |

### `Quote`
Dataclass: `direction`, `price`, `size`, `level`. Returned as `List[Quote]` from `compute_quotes()`.

### `Quoter`

| Method | Description |
|---|---|
| `compute_quotes(step, t)` | Returns `(List[Quote], List[int])` — 20 quotes + IDs to cancel |
| `update_live_ids(ids)` | Register submitted order IDs for cancellation next step |
| `update_inventory(delta_q)` | Update EUR inventory after a fill |
| `record_fill(step, t, direction, price, size, delta)` | Store fill in history for Phase 2 calibration |
| `needs_hedge()` | True if delta limit breached |
| `hedge_order(depth_B, depth_C, fair_mid)` | Returns `(size_B, size_C, fee_cost)` — dynamic routing |
| `fill_cost(size, fair_mid)` | Maker fee on an Exchange A fill |
| `snapshot(step, t)` | Full dict of intermediate quantities for backtesting report |

---

## Usage

```python
from utils.stock_simulation import Stock
from utils.market_simulator import Market
from utils.market_maker.quoter import Quoter, QuoterConfig

stock = Stock(drift=0.0, vol=0.10, origin=1.10, tick_size=0.0001)
stock.simulate_gbm(n_days=30)

market_B = Market(stock)
market_B.generate_noised_mid_price(vol_factor=0.05)
market_B.build_spread()

market_C = Market(stock)
market_C.generate_noised_mid_price(vol_factor=0.05)
market_C.build_spread()

quoter = Quoter(market_B, market_C, capital_K=1_000_000.0)

for step, t in enumerate(stock._time_grid):
    quotes, cancel_ids = quoter.compute_quotes(step, t)
    # cancel cancel_ids on Exchange A's order book
    # submit quotes as limit orders → get back new_ids
    quoter.update_live_ids(new_ids)

    # on fill:
    quoter.update_inventory(delta_q)
    quoter.record_fill(step, t, direction, price, size, delta)
    pnl -= quoter.fill_cost(size, fair_mid)

    # on delta breach:
    if quoter.needs_hedge():
        size_B, size_C, fee_cost = quoter.hedge_order(depth_B, depth_C, fair_mid)
        pnl -= fee_cost
```

---

## Reference

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
