# market_maker — Quoting Framework

## Context

Exchange A is a new venue that has approached our firm to act as Designated Market Maker. In Phase 1, the exchange has zero trading history — all quoting logic must be purely heuristic. We are the sole market maker on Exchange A, with two reference markets:

- **Exchange B** — 75% of total EUR/USD volume, data arrives with 200ms latency
- **Exchange C** — 25% of total EUR/USD volume, data arrives with 170ms latency

HFTs access both feeds in 50ms, giving them a 150ms/120ms edge over us.

Starting capital K is half in EUR, half in USD (flat inventory = 0 at session start). Delta risk limit is 90%: when `|inventory| / K > 90%` we hedge back to flat (inventory = 0) via taker market orders on B and/or C.

EUR/USD trades 24 hours a day. The quoter organises the day into three FX sessions with hard resets at **08:00 and 16:00 UTC** (London, New York). Each session uses an independent horizon `T = 8h` for the Avellaneda-Stoikov urgency model.

---

## Architecture — Producer/Listener Pattern

```
Quoter (Producer)                    Order_book (Listener)
─────────────────                    ─────────────────────
compute_quotes()  ──► cancel_ids ──► cancel_orders()
                  ──► new quotes ──► post_mm_quotes()

                  ◄── FillEvent  ◄── _fire_fill()   [on each match]
on_fill()        ◄──────────────
```

The `Order_book` fires a `FillEvent` to the `Quoter` on every fill (full or partial). The `Quoter`:
- Always updates `inventory` and logs the fill in `_fill_history`
- Full fill → queues `_pending_fills` → re-emits that (direction, level) slot next step
- Partial fill → queues `_pending_topups` → posts a complementary top-up order at the same price for the gap size

---

## Quoting Framework — Step by Step

### Step 1 — Fair mid price

Prices from B and C are read with latency offsets applied, then the cross-venue best is taken:

```
best_bid = max(bid_B(t − 200ms), bid_C(t − 170ms))
best_ask = min(ask_B(t − 200ms), ask_C(t − 170ms))
fair_mid = (best_bid + best_ask) / 2
```

B and C each support 5 spread types: Static, Stochastic, Adaptive, Asymmetric, Skewed. The Quoter reads the active spread via the `market.bid_price` / `market.ask_price` properties — no hardcoding.

### Step 2 — Adaptive volatility (EWMA)

Before `vol_window` steps have elapsed the quoter falls back to the parametric `stock.vol`. Afterwards it uses EWMA realized vol with a safety floor at 20% of parametric vol.

### Step 3 — Reservation price (Avellaneda-Stoikov + Guéant terminal penalty)

```
penalty_factor = 1 + strength × (t/T)³
r(t) = fair_mid − q × γ × σ² × (T−t) × penalty_factor
```

The cubic penalty ramps hard in the final ~20% of the session to ensure clean end-of-session inventory.

### Step 4 — Order flow imbalance tilt (Cartea & Jaimungal)

```
imbalance = (n_ask_hits − n_bid_hits) / total    ∈ [−1, 1]
r(t) += alpha_imbalance × imbalance × fair_mid
```

When clients predominantly buy from us (hit our ask), the reservation price shifts up. Resets at each FX session boundary.

### Step 5 — Total spread (three components)

```
spread_AS        = [γ × (σ×100)² × (T−t) + (2/γ) × ln(1 + γ/k)] / 10000 × fair_mid
spread_latency   = 2 × (σ / sqrt(T_year)) × sqrt(effective_gap_s)
spread_inventory = α_spread × (q/K)² × spread_AS
total_spread     = spread_AS + spread_latency + spread_inventory
```

### Step 6 — Asymmetric half-spreads (Guéant)

```
skew_delta = q × sqrt(γσ²/(2k))
ask_half   = half_spread − skew_delta
bid_half   = half_spread + skew_delta
```

### Step 7 — 10-level ladder with inventory-skewed sizing

Prices step one tick further from mid at each level. Sizes decay exponentially and are skewed against inventory:
```
bid_size_i = round(Q_base × exp(−β×i) × (1 − 0.5 × clip(q/K, −1, 1)))
ask_size_i = round(Q_base × exp(−β×i) × (1 + 0.5 × clip(q/K, −1, 1)))
```

### Step 8 — Selective requote (priority order)

| Priority | Trigger | Action |
|---|---|---|
| 1 | First call ever | Full 20-order ladder, no cancels |
| 2 | FX session boundary | Cancel all, full ladder |
| 3 | Best price drifted > threshold | Cancel all, full ladder |
| 4 | Full fill + partial top-ups pending | Requote filled slots only + top-up orders |
| 5 | Stale orders + stressed inventory | Cancel stale slots, reprice those only |
| — | Nothing triggered | No action |

Partial fill top-up: the surviving partial order stays resting (no cancel). A complementary order is posted at the **same price** for `original_size − remaining_size`.

### Step 9 — Dynamic hedge routing

When `|q| / K > 90%`, target is flat (inventory = 0). Hedge is executed as taker market orders on B and C.

**Depth**: read from `market_B.depth[step − lag_B]` and `market_C.depth[step − lag_C]` — the actual EUR size quoted at the best price on each venue at the lagged step, same latency as price feeds. If depth arrays were not generated, falls back to `capital_K × weight_venue`.

**Routing score** (higher = preferred):
```
vol_per_s    = σ / sqrt(T_year)
latency_cost = vol_per_s × latency_venue_s   (expected adverse move during execution lag)
impact_cost  = fair_mid × 0.0001 / depth     (Kyle-lambda market impact)

score_venue  = 1 / (taker_fee + latency_cost + impact_cost)
ratio_B      = score_B / (score_B + score_C)
```
C's shorter latency (170ms vs 200ms) gives it a higher score when vol is elevated. Total hedge is capped at `depth_B + depth_C`; if B's allocation exceeds its depth, the overflow is rerouted to C.

**Partial hedge and emergency fallback**:

| Post-hedge `|q|/K` | Outcome |
|---|---|
| ≤ 80% (`hedge_partial_limit`) | Hedge succeeded (full or partial). Normal quoting resumes next step. |
| > 80% | Available depth on B+C was insufficient. Sets `_hedge_emergency = True`. |

When `_hedge_emergency` is active, `compute_quotes` multiplies the A-S penalty factor by `emergency_penalty_multiplier` (default 5×), forcing an extreme reservation price shift away from fair mid on the inventory side. This makes our quotes on A drastically asymmetric — aggressively attracting client flow in the reducing direction until inventory naturally recovers below 90%.

Each hedge leg is recorded in `_fill_history` with `is_hedge=True` and `venue="B"` or `"C"`.

---

## Trade History

`Quoter.trade_history` returns a DataFrame with one row per fill (MM fills + hedge legs):

| Column | Description |
|---|---|
| `order_id` | ID of the MM order that was matched (None for hedges) |
| `level` | Ladder level (1 = best quote, 0 for hedges) |
| `step` | Simulation step |
| `t` | Elapsed seconds |
| `direction` | `"buy"` or `"sell"` (MM perspective) |
| `price` | Execution price |
| `size` | Matched size (EUR) |
| `is_full_fill` | False if order is still partially resting |
| `fair_mid` | Weighted reference fair mid at time of fill |
| `fee_cost` | Absolute fee paid (USD) |
| `cash_flow` | Signed cash impact net of fees (USD) |
| `inventory_after` | Cumulative EUR inventory after this fill |
| `is_hedge` | True for hedge legs on B/C |
| `venue` | `"A"`, `"B"`, or `"C"` |

**Cash flow convention (USD):**
- MM sells EUR (client buys): `+price × size × (1 − maker_fee_A)`
- MM buys EUR (client sells): `−price × size × (1 + maker_fee_A)`
- Hedge sell on B/C: `+bid_venue × size × (1 − taker_fee)`  (crosses the bid as a taker)
- Hedge buy on B/C: `−ask_venue × size × (1 + taker_fee)`  (crosses the ask as a taker)

Use `PnLTracker` (see `pnl_tracker_README.md`) to decompose this into realized P&L, MtM P&L, inception spread, and inventory revaluation.

---

## Classes and Files

| File | Contents |
|---|---|
| `quoter.py` | `Quoter`, `QuoterConfig` |
| `pnl_tracker.py` | `PnLTracker` — static P&L analysis methods |
| `events.py` (order_book/) | `FillEvent` dataclass shared between book and quoter |

### `QuoterConfig` parameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.1 | Risk-aversion coefficient |
| `k` | 1.5 | Order arrival decay rate |
| `T` | `8 × 3600` | Session horizon (one FX session) |
| `alpha_spread` | 0.5 | Inventory spread widening coefficient |
| `use_asymmetric_delta` | True | Guéant asymmetric half-spreads |
| `terminal_penalty_strength` | 5.0 | End-of-session inventory urgency |
| `n_levels` | 10 | Price levels each side |
| `beta` | 0.3 | Size decay across levels |
| `Q_base` | 100,000 | Base EUR size at best level |
| `tick_size` | 0.0001 | 1 bp tick |
| `requote_threshold_spread_fraction` | 0.25 | Min price move (fraction of spread) to trigger requote |
| `stale_steps` | 300 | Age threshold for staleness rule |
| `stale_inventory_fraction` | 0.5 | Inventory stress threshold for staleness rule |
| `imbalance_window` | 50 | Rolling window for OFI signal |
| `alpha_imbalance` | 0.0002 | OFI tilt scaling coefficient |
| `vol_window` | 6000 | EWMA vol estimation window (steps) |
| `weight_B / weight_C` | 0.75 / 0.25 | Volume weights for fair mid |
| `latency_B_s / latency_C_s` | 0.200 / 0.170 | Data latency (seconds) |
| `latency_hft_s` | 0.050 | HFT latency |
| `delta_limit` | 0.90 | Hedge trigger (fraction of capital) |
| `hedge_partial_limit` | 0.80 | If post-hedge ratio still above this, emergency mode activates |
| `emergency_penalty_multiplier` | 5.0 | A-S penalty amplifier during emergency (skews quotes on A) |
| `fee_A_maker` | 0.0001 | Exchange A maker fee |
| `fee_A_taker` | 0.0004 | Exchange A taker fee |
| `fee_B_maker` | 0.00009 | Exchange B maker fee |
| `fee_B_taker` | 0.0002 | Exchange B taker fee |
| `fee_C_maker` | 0.00009 | Exchange C maker fee |
| `fee_C_taker` | 0.0003 | Exchange C taker fee |

### `Quoter` public methods

| Method | Description |
|---|---|
| `compute_quotes(step, t, resting_orders)` | Returns `(List[Order], List[str])` — new quotes + IDs to cancel |
| `on_fill(event)` | Callback registered with `Order_book` — updates inventory, logs fill, queues re-quote or top-up |
| `execute_hedge(step, t, fair_mid)` | Executes hedge if `needs_hedge()`, records legs in trade history. Returns True if hedge fired |
| `needs_hedge()` | True if `|inventory| / capital_K > delta_limit` |
| `hedge_order(depth_B, depth_C, fair_mid, sigma)` | Returns `(size_B, size_C, fee_cost)` — optimal routing split capped at available depth |
| `trade_history` | Property — returns full fill log as DataFrame (14 columns) |
| `snapshot(step, t)` | Full dict of intermediate quoting quantities for diagnostics |

---

## Minimal usage

```python
import copy
from utils.stock_simulation import Stock
from utils.market_simulator.market import Market
from utils.order_book.order_book_impl import Order_book
from utils.market_maker.quoter import Quoter, QuoterConfig
from utils.market_maker.pnl_tracker import PnLTracker

stock = Stock(drift=0.0, vol=0.20)
stock.simulate_garch(n_days=1, dt_seconds=0.01)

market_B = Market(stock)
market_B.generate_noised_mid_price()
market_B.build_spread(option="Skew", window_size=600, alpha=0.5, gamma=0.3, ema_span=500, threshold=3)
market_B.generate_depth(mean_eur=500_000)   # EUR size at best quote each step

market_C = copy.deepcopy(market_B)
market_C.build_spread(option="Adaptive", window_size=600)
market_C.generate_depth(mean_eur=200_000)   # C has less volume than B

book = Order_book()
mm = Quoter(market_B, market_C, config=QuoterConfig(), capital_K=1_000_000.0)
book.register_quoter_listener(mm.on_fill)

dt = stock.time_step
for step in range(stock.n_steps):
    t = step * dt
    book.tick(step)
    quotes, cancels = mm.compute_quotes(step, t, book.mm_resting_orders)
    book.cancel_orders(cancels)
    book.post_mm_quotes(quotes)
    # ... route client orders ...
    mm.execute_hedge(step, t, fair_mid)

rep = PnLTracker.report(mm.trade_history, current_mid=fair_mid)
print(rep)
```

---

## References

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.

Guéant, O. (2017). *Optimal market making*. Applied Mathematical Finance, 24(2), 112–154.

Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
