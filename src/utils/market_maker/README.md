# market_maker — Quoting Framework

## Context

Exchange A is a new venue that has approached our firm to act as Designated Market Maker. In Phase 1, the exchange has zero trading history — all quoting logic must be purely heuristic. We are the sole market maker on Exchange A, with two reference markets:

- **Exchange B** — 75% of total EUR/USD volume, data arrives with 200ms latency
- **Exchange C** — 25% of total EUR/USD volume, data arrives with 170ms latency

HFTs access both feeds in 50ms, giving them a 150ms/120ms edge over us.

EUR/USD trades 24 hours a day, 5 days a week. The quoter organises this into three FX sessions with hard resets at **00:00, 08:00 and 16:00 UTC** (Tokyo, London, New York). Each session is treated as an independent horizon `T = 8h` for the AS/Guéant urgency model.

---

## Architecture — Producer/Listener Pattern

The quoter and order book interact through an event-driven loop rather than a simple function call:

```
Quoter (Producer)                    Order_book (Listener)
─────────────────                    ─────────────────────
compute_quotes()  ──► cancel_ids ──► cancel_orders()
                  ──► new quotes ──► add_order() × n

                  ◄── FillEvent  ◄── _fire_fill()  [on each match]
on_fill()        ◄──────────────
```

The `Order_book` fires a `FillEvent` to the `Quoter` every time one of its resting orders is matched (fully or partially). The `Quoter` uses this to update inventory immediately and — for full fills only — to force a requote of that side on the next step.

---

## Quoting Framework — Step by Step

### Step 1 — Fair mid price

We observe prices from B and C with their respective latency offsets applied, then take the cross-venue best:

```
best_bid = max(bid_B(t − 200ms), bid_C(t − 170ms))
best_ask = min(ask_B(t − 200ms), ask_C(t − 170ms))
fair_mid = (best_bid + best_ask) / 2
```

### Step 2 — Adaptive volatility (EWMA)

Before `vol_window` steps have elapsed the quoter falls back to the parametric `stock.vol`. Once enough history exists it switches to EWMA realized vol:

```
log_rets = diff(log(noised_mid_price[step - vol_window : step]))
ewma_vol = ewm(span=vol_window).std() / sqrt(dt / T_year)
```

A safety floor at 20% of parametric vol prevents degenerate near-zero spreads during quiet periods.

### Step 3 — Reservation price (Avellaneda-Stoikov + Guéant terminal penalty)

```
penalty_factor = 1 + strength × (t/T)³
r(t) = fair_mid − q × γ × σ² × (T−t) × penalty_factor
```

The cubic penalty is negligible for most of the session and ramps hard in the final ~20% to ensure clean end-of-session inventory. At `t=0` it equals 1 (plain AS); at `t=T` it equals `1 + strength`.

### Step 4 — Order flow imbalance tilt (Cartea & Jaimungal)

Client order flow carries an informational signal: if clients are predominantly hitting our ask (buying from us), the price is likely drifting up — informed buyers are taking liquidity. We tilt the reservation price in the direction of dominant flow:

```
imbalance = (n_ask_hits − n_bid_hits) / (n_ask_hits + n_bid_hits)   ∈ [−1, 1]
reservation_price += alpha_imbalance × imbalance × fair_mid
```

When clients buy from us (hit our ask),  → reservation price shifts up → both bid and ask shift up, making it more expensive to buy from us and cheaper to sell to us. This is the informed flow response described in Cartea & Jaimungal (2015).

The signal is computed over a rolling window of the last  fills and resets at each FX session boundary.

### Step 5 — Total spread (three components)

**Component A — Avellaneda-Stoikov base spread**
```
spread_AS = [γ × (σ×100)² × (T−t) + (2/γ) × ln(1 + γ/k)] / 10000 × fair_mid
```

**Component B — Latency premium**

We are 142.5ms behind HFTs on average (0.75×150ms + 0.25×120ms). We charge for this adverse selection window:
```
spread_latency = 2 × (σ / sqrt(T_year)) × sqrt(142.5ms)
```

**Component C — Inventory-dependent spread widening [Guéant fix 1]**

Guéant (2017) shows the optimal spread is convex in inventory. We approximate:
```
spread_inventory = α × (q/K)² × spread_AS
```
Always widens — never narrows — the spread when inventory is non-zero.

### Step 6 — Asymmetric half-spreads [Guéant fix 2]

Standard AS uses equal half-spreads. Guéant shows the optimal skew is:
```
skew_delta = q × sqrt(γσ²/(2k))
ask_half   = half_spread − skew_delta   # narrows when long → attracts sellers
bid_half   = half_spread + skew_delta   # widens when long → repels buyers
```
This is complementary to the reservation price shift: the reservation price moves the *centre* of the spread; the asymmetric delta changes its *shape*.

### Step 7 — 10-level ladder with inventory-skewed sizing

Prices step one tick further from mid at each level. Sizes are skewed to reinforce mean-reversion:
```
inventory_skew = clip(q/K, −1, 1)
bid_size_i = Q_base × exp(−β×i) × (1 − 0.5 × skew)
ask_size_i = Q_base × exp(−β×i) × (1 + 0.5 × skew)
```

### Step 8 — Selective requote (three rules)

Rather than cancel-and-replace the full ladder every step, we only cancel individual resting orders when one of the following fires:

| Rule | Trigger |
|---|---|
| **Fill** | Order fully consumed — requote that side immediately |
| **Threshold** | `\|resting_price − theo_price\| > fraction × total_spread` |
| **Staleness** | Order age > `stale_steps` AND `\|q/K\| > stale_inventory_fraction` |

Partially filled orders are **not** force-cancelled — they remain resting and are subject to the threshold and staleness rules only.

At each **FX session boundary** (00:00, 08:00, 16:00 UTC) all resting orders are cancelled regardless, and the session resets cleanly.

### Step 9 — Dynamic hedge routing

When `|q| / K > 90%`, we send market orders on B and/or C. The split is fee- and depth-aware:
```
score_venue = 1 / (taker_fee + impact_factor / depth)
ratio_B     = score_B / (score_B + score_C)
```

---

## Classes and Files

### `events.py`
Contains `FillEvent` — the shared dataclass imported by both `order_book_impl.py` and `quoter.py` to avoid circular dependencies.

| Field | Type | Description |
|---|---|---|
| `order_id` | str | ID of the MM order that was matched |
| `direction` | str | `"buy"` or `"sell"` — the MM order side |
| `price` | float | Execution price |
| `size` | float | Matched portion (may be < original size) |
| `step` | int | Simulation step at time of fill |
| `is_full_fill` | bool | `False` if order is still partially resting |

### `QuoterConfig`

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.1 | Risk-aversion coefficient |
| `k` | 1.5 | Order arrival decay rate |
| `T` | `8 × 3600` | Session horizon (one FX session) |
| `alpha_spread` | 0.5 | Inventory spread widening coefficient |
| `use_asymmetric_delta` | True | Enable Guéant asymmetric half-spreads |
| `terminal_penalty_strength` | 5.0 | End-of-session inventory urgency |
| `n_levels` | 10 | Price levels each side |
| `beta` | 0.3 | Size decay across levels |
| `Q_base` | 100,000 | Base EUR size at best level |
| `tick_size` | 0.0001 | 1bp tick |
| `requote_threshold_spread_fraction` | 0.25 | Min price move (as fraction of spread) to trigger requote |
| `stale_steps` | 300 | Steps before a stressed order is force-requoted |
| `stale_inventory_fraction` | 0.5 | Inventory stress threshold for staleness rule |
| `imbalance_window` | 50 | Number of recent fills used to compute the flow signal |
| `alpha_imbalance` | 0.0002 | Scaling coefficient for the imbalance reservation price tilt |
| `vol_window` | 6000 | EWMA span in steps |
| `weight_B/C` | 0.75/0.25 | Volume weights |
| `latency_B/C_s` | 0.200/0.170 | Data latency in seconds |
| `latency_hft_s` | 0.050 | HFT latency |
| `delta_limit` | 0.90 | Hedge trigger threshold |
| `fee_A_maker` | 0.0001 | Exchange A maker fee |
| `fee_A_taker` | 0.0004 | Exchange A taker fee |
| `fee_B_maker` | 0.00009 | Exchange B maker fee |
| `fee_B_taker` | 0.0002 | Exchange B taker fee |
| `fee_C_maker` | 0.00009 | Exchange C maker fee |
| `fee_C_taker` | 0.0003 | Exchange C taker fee |

### `Quote`
Dataclass: `direction`, `price`, `size`, `level`. Returned as `List[Quote]` from `compute_quotes()`.

### `Quoter`

| Method | Description |
|---|---|
| `compute_quotes(step, t, resting_orders)` | Returns `(List[Quote], List[str])` — new quotes + IDs to cancel |
| `on_fill(event)` | Callback registered with the OrderBook — updates inventory and queues forced requote on full fills |
| `needs_hedge()` | True if delta limit breached |
| `hedge_order(depth_B, depth_C, fair_mid)` | Returns `(size_B, size_C, fee_cost)` — dynamic routing |
| `fill_cost(size, fair_mid)` | Maker fee on an Exchange A fill |
| `snapshot(step, t)` | Full dict of intermediate quantities for backtesting report — includes `imbalance` field |

---

## Usage

```python
from utils.stock_simulation import Stock
from utils.market_simulator import Market
from utils.market_maker.quoter import Quoter, QuoterConfig

stock = Stock(drift=0.0, vol=0.10, origin=1.10, tick_size=0.0001)
stock.simulate_gbm(n_days=5)

market_B = Market(stock)
market_B.generate_noised_mid_price(vol_factor=0.05)
market_B.build_spread()

market_C = Market(stock)
market_C.generate_noised_mid_price(vol_factor=0.05)
market_C.build_spread()

quoter = Quoter(market_B, market_C, capital_K=1_000_000.0)

# Wire the fill callback — OrderBook will notify Quoter on every match
book.register_quoter_listener(quoter.on_fill)

for step, t in enumerate(stock._time_grid):
    book.tick(step)

    quotes, cancel_ids = quoter.compute_quotes(step, t, book.mm_resting_orders)
    book.cancel_orders(cancel_ids)

    for q in quotes:
        order = Order(_generate_order_id(), q.direction, q.price, q.size, "limit_order")
        book.add_order(order)

    # on delta breach:
    if quoter.needs_hedge():
        size_B, size_C, fee_cost = quoter.hedge_order(depth_B, depth_C, fair_mid)
        pnl -= fee_cost
```

---

## References

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.

Guéant, O. (2017). *Optimal market making*. Applied Mathematical Finance, 24(2), 112–154.

Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
