# market_maker — Quoting Framework

## Context

Exchange A is a new venue that has approached our firm to act as Designated Market Maker. In Phase 1, the exchange has zero trading history — all quoting logic must be purely heuristic. We are the sole market maker on Exchange A, with two reference markets:

- **Exchange B** — 75% of total EUR/USD volume, data arrives with 200ms latency
- **Exchange C** — 25% of total EUR/USD volume, data arrives with 170ms latency

HFTs access both feeds in 50ms, giving them a 150ms/120ms edge over us.

Starting capital K is half in EUR, half in USD (flat inventory = 0 at session start). Capital is thus split 50/50 between EUR and USD at inception, so risk limits are expressed against `K / 2` (each currency leg). Delta risk limit fires when `|inventory| > delta_limit × (K/2)` **or** `|inventory × fair_mid| > delta_limit × (K/2)` — both EUR and USD notional are checked. We then hedge via taker market orders on B and/or C.

EUR/USD trades 24 hours a day. The quoter organises the day into three FX sessions (Tokyo, London, New York) with hard resets at **08:00 and 16:00 UTC**. The A-S inventory penalty uses an infinite-horizon stationary discount `ω = 1/(8h)` — the effective risk horizon — rather than a finite `T − t` countdown, because a 24/5 FX market has no terminal liquidation date.

> **Session-adaptive k — currently disabled.** `_SESSION_K_MULTIPLIERS` in [quoter.py:21-25](quoter.py#L21-L25) originally scaled `cfg.k` by session liquidity (Tokyo 0.75×, London 1.25×, NY 1.00×). All three multipliers are presently set to `1.00` (the intended values are preserved in comments), so `_session_k(t)` and `_precomp_k` return a flat `cfg.k` across sessions. Re-enable by uncommenting the Tokyo/London factors.

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

Before `vol_min_s` seconds of data have accumulated, the quoter falls back to the parametric `stock.vol`. Afterwards it uses EWMA realized vol over a `vol_window_s` rolling window. The full per-step EWMA path is pre-computed once at `Quoter.__init__` for O(1) per-step lookup.

### Step 3 — Reservation price (Avellaneda-Stoikov, infinite horizon)

```
ρ = q / K
r(t) = fair_mid − ρ × half_spread
```

The original A-S shift `q × γ × σ² × (T−t)` is rescaled to the inventory ratio × half-spread. Motivation: applying the raw formula with q in EUR (0–900k) produces a shift of several hundred bps at 10% inventory, three orders of magnitude larger than the half-spread itself. Rescaling pins the shift to the spread, so at `|ρ| = 1` the quoted centre is displaced by exactly one half-spread — direction preserved, magnitude calibrated, no new free parameter.

### Step 4 — Order flow imbalance tilt (Cartea & Jaimungal)

```
imbalance = (n_ask_hits − n_bid_hits) / total    ∈ [−1, 1]
r(t) += alpha_imbalance × imbalance × fair_mid
```

When clients predominantly buy from us (hit our ask), the reservation price shifts up. Resets at each FX session boundary.

### Step 5 — Total spread (three components + fee floor)

```
spread_AS        = [γ × (σ×100)² × (1/ω) + (2/γ) × ln(1 + γ/k)] / 10000 × fair_mid
spread_latency   = 2 × (σ / sqrt(T_year)) × sqrt(effective_gap_s)
spread_inventory = α_spread × (q/K)² × spread_AS
total_spread     = spread_AS + spread_latency + spread_inventory
fee_floor        = min_edge_multiple × fee_A_maker × fair_mid
half_spread      = max(total_spread / 2, tick_size, fee_floor)
```

The fee floor guarantees the inception half-spread covers the maker fee: no fill is ever gross-negative. It binds in calm regimes where the three adversarial premia would otherwise collapse below fees.

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
| 2 | FX session boundary (08:00 / 16:00 UTC) | Cancel all, full ladder |
| 3 | Theoretical best drifted > `requote_threshold_spread_fraction` × spread | Cancel all, full ladder |
| 4 | Full fill + partial top-ups pending | Requote filled slots only + top-up orders |
| 5 | `|Δ inventory| / K > inventory_requote_fraction` since last reprice | Cancel all, full ladder |
| 6 | Any order older than `stale_s` seconds | Cancel stale slots, reprice at `stale_tight_fraction × spread` |
| — | Nothing triggered | No action |

Priorities 3 and 5 close two distinct leaks: P3 fires when *market prices* have moved, P5 fires when *our inventory* has moved while prices are flat (silent accumulation at a stale skew). P6 fires on pure staleness and uses a tighter spread so the refreshed quotes compete back into relevance.

Partial fill top-up (P4): the surviving partial order stays resting (no cancel). A complementary order is posted at the **same price** for `original_size − remaining_size`.

### Step 9 — Dynamic hedge routing

When either `|q| > delta_limit × (K/2)` or `|q × fair_mid| > delta_limit × (K/2)` (i.e. the EUR leg *or* its USD notional breaches the delta limit on half-capital), the target is flat (inventory = 0). Hedge is executed as taker market orders on B and C. Fair mid for the USD valuation is the cross-venue B/C mid, not `mid_A` — A's mid is polluted by our own skewed quotes.

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

| Post-hedge `|q|` | Outcome |
|---|---|
| ≤ `hedge_partial_limit × (K/2)` | Hedge succeeded (full or partial). Normal quoting resumes next step. |
| >  `hedge_partial_limit × (K/2)` | Available depth on B+C was insufficient. Sets `_hedge_emergency = True`. |

When `_hedge_emergency` is active, `compute_quotes` multiplies the A-S inventory-risk term by `emergency_penalty_multiplier` (default 5×), forcing an extreme reservation price shift away from fair mid on the inventory side. This makes our quotes on A drastically asymmetric — aggressively attracting client flow in the reducing direction until inventory naturally recovers below the limit.

Each hedge leg is recorded in `_fill_history` with `is_hedge=True` and `venue="B"` or `"C"`.

**EOD flat.** If `eod_flat_interval > 0`, `execute_hedge` also force-flattens inventory to zero every `eod_flat_interval` seconds (tracked via `_last_eod_day`), regardless of whether `needs_hedge` would have fired. These legs are tagged `is_eod_flat=True`. Default is `0.0` (disabled — continuous run).

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
| `is_eod_flat` | True if this hedge leg was fired by the EOD flat trigger |
| `venue` | `"A"`, `"B"`, or `"C"` |

**Cash flow convention (USD):**
- MM sells EUR (client buys): `+price × size × (1 − maker_fee_A)`
- MM buys EUR (client sells): `−price × size × (1 + maker_fee_A)`
- Hedge sell on B/C: `+bid_venue × size × (1 − taker_fee)`  (crosses the bid as a taker)
- Hedge buy on B/C: `−ask_venue × size × (1 + taker_fee)`  (crosses the ask as a taker)

Use `PnLTracker` (lives in [`src/utils/report/`](../report/), not this folder) to decompose this into realized P&L, MtM P&L, inception spread, and inventory revaluation.

---

## Classes and Files

| File | Contents |
|---|---|
| [quoter.py](quoter.py) | `Quoter`, `QuoterConfig` |
| [events.py](../order_book/events.py) (in `order_book/`) | `FillEvent` dataclass shared between book and quoter |

`PnLTracker` has moved to [`src/utils/report/`](../report/) — import it from there.

### `QuoterConfig` parameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.1 | Risk-aversion coefficient |
| `k` | 0.3 | Order arrival decay rate |
| `omega` | `1 / (8 × 3600)` | Infinite-horizon discount rate (risk horizon ≈ one FX session) |
| `alpha_spread` | 0.5 | Inventory spread widening coefficient |
| `use_asymmetric_delta` | True | Guéant asymmetric half-spreads |
| `n_levels` | 10 | Price levels each side |
| `beta` | 0.3 | Size decay across levels |
| `Q_base` | 70,000 | Base EUR size at best level |
| `tick_size` | 0.0001 | 1 bp tick |
| `requote_threshold_spread_fraction` | 0.25 | Min price move (fraction of spread) to trigger P3 requote |
| `stale_s` | 10.0 | Age threshold for staleness rule (seconds) |
| `stale_tight_fraction` | 0.7 | Spread multiplier when P6 refreshes stale orders |
| `inventory_requote_fraction` | 0.05 | \|Δ inventory\| / K threshold to trigger P5 requote |
| `imbalance_min_samples` | 10 | Minimum fills before the imbalance signal is trusted |
| `imbalance_window` | 50 | Rolling window for OFI signal |
| `alpha_imbalance` | 0.0002 | OFI tilt scaling coefficient |
| `vol_window_s` | 60.0 | EWMA vol estimation lookback (seconds) |
| `vol_min_s` | 2.0 | Warm-up before switching to realized vol (seconds) |
| `weight_B / weight_C` | 0.75 / 0.25 | Volume weights for fair mid |
| `latency_B_s / latency_C_s` | 0.200 / 0.170 | Data latency (seconds) |
| `latency_hft_s` | 0.050 | HFT latency |
| `delta_limit` | 0.90 | Hedge trigger (fraction of `K/2`; applied independently to EUR and USD legs) |
| `hedge_partial_limit` | 0.80 | If post-hedge \|q\| > `hedge_partial_limit × (K/2)`, emergency mode activates |
| `emergency_penalty_multiplier` | 5.0 | A-S penalty amplifier during emergency (skews quotes on A) |
| `eod_flat_interval` | 0.0 | EOD-flat cadence in seconds; `0.0` disables continuous-running EOD flats |
| `min_edge_multiple` | 1.0 | Inception spread floor as a multiple of `fee_A_maker × fair_mid` (≥ 1 guarantees gross-positive fills) |
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
| `execute_hedge(step, t)` | Executes hedge if `needs_hedge(fair_mid)`, records legs in trade history. Fair mid is computed internally from the cached B/C quotes set during `compute_quotes`. Returns True if hedge fired |
| `needs_hedge(fair_mid=1.0)` | True if either `|inventory| > delta_limit × (K/2)` or `|inventory × fair_mid| > delta_limit × (K/2)` |
| `hedge_order(depth_B, depth_C, fair_mid, sigma)` | Returns `(size_B, size_C, fee_cost)` — optimal routing split capped at available depth |
| `trade_history` | Property — returns full fill log as DataFrame (14 columns) |
| `snapshot(step, t)` | Full dict of intermediate quoting quantities for diagnostics |

---

## Minimal usage

The canonical driver is `Controller` ([`src/utils/report/`](../report/)) — **do not write bare step-loops**.

```python
from utils.market_maker.quoter import Quoter, QuoterConfig
from utils.order_book.order_book_impl import Order_book
from utils.client_flow import ClientFlowGenerator
from utils.report import Controller

book  = Order_book()
mm    = Quoter(market_B, market_C, config=QuoterConfig(), capital_K=1_000_000.0)
book.register_quoter_listener(mm.on_fill)
gen   = ClientFlowGenerator(seed=42)

ctrl  = Controller(market_B, market_C, book, mm, gen.generate_step)
ctrl.simulate()
ctrl.report()
```

See [test/report.ipynb](../../../test/report.ipynb) for the full reporting notebook.

---

## References

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.

Guéant, O. (2017). *Optimal market making*. Applied Mathematical Finance, 24(2), 112–154.

Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
