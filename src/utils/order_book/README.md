# `order_book` — Exchange A Limit Order Book

DataFrame-backed LOB with price-time matching, partial fills, and a producer/listener pattern for fill notifications. Exchange A is the venue the firm market-makes on; client flow and HFT quotes route through this book.

---

## Files

| File | Role |
|---|---|
| [order_book_impl.py](order_book_impl.py) | `Order`, `Order_book`, `generate_order_id` |
| [events.py](events.py) | `FillEvent` dataclass — shared with Quoter / HFT agent |
| [graphic_utils.py](graphic_utils.py) | `plot_order_book()` — cumulative depth chart |
| [config.py](config.py) | Constants: `ORDER_DIRECTION`, `ORDER_TYPE` (only `"limit_order"` enabled) |
| [utils.py](utils.py) | Comment-only: documents the intraday time grid convention (`dt = 50 ms`) |

---

## `Order`

Value-object style. Fields are stored as underscored attributes (`_id`, `_direction`, ...) and should not be mutated after creation — to top-up a partially filled order, post a **new** order at the same price for the gap.

| Field | Description |
|---|---|
| `id` | Unique order ID (monotonic counter; use `generate_order_id()`) |
| `direction` | `"buy"` or `"sell"` |
| `price` | Float |
| `size` | Integer (EUR) |
| `type` | `"limit_order"` (market orders are disabled) |
| `origin` | `"market_maker"`, `"client"`, or `"hft"` |
| `level` | Ladder level (1 = best, 10 = deepest; `0` for non-MM orders) |

---

## `FillEvent`

Fired by `Order_book` on every match (full or partial). Consumed by the Quoter's `on_fill` and the HFT agent's `on_hft_fill`.

```python
@dataclass
class FillEvent:
    order_id:     str
    direction:    str      # direction of the matched MM/HFT order
    price:        float
    size:         float    # matched portion only
    step:         int
    level:        int
    is_full_fill: bool     # False = partial, order still resting
```

---

## `Order_book`

### Storage

Separate dicts for matching speed; DataFrame views exist for display only.

| Attribute | Purpose |
|---|---|
| `_orders` | MM **and** HFT orders (compete in matching). `{id → {direction, price, size, origin, level, seq}}` |
| `_client_orders` | Client orders only |
| `_mm_resting` | Registry for MM orders only (not HFT): tracks `original_size`, `remaining_size`, `post_step`, `level`. Used by Quoter for partial top-ups and Priority-6 staleness scans. |
| `_sorted_mm_asks` / `_sorted_mm_bids` | Sorted caches, rebuilt only when `_mm_dirty=True` |
| `_match_log` | List of match dicts → `_df_matches` on demand |
| `_submission_log` | Every order routed through the book (opt-in via `track_submissions=True`) → `order_history` on demand |

DataFrame views (`_df_order_book`, `_df_bid_book`, `_df_ask_book`, `_df_matches`, `order_history`) are rebuilt on each access and exist only for notebook inspection.

### Producer / listener pattern

```
Quoter.compute_quotes ──► post_mm_quotes ──► add_order (origin="market_maker")
HFTAgent.step         ──► add_order       (origin="hft")
ClientFlow            ──► add_order       (origin="client")
                          ▼
                      try_clear  ─── matches ─── _fire_fill ──► FillEvent ──►┐
                                                                             │
                                         on_fill (Quoter) ◄──────────────────┤
                                         on_hft_fill (HFT) ◄─────────────────┘
                                         (dispatched by origin)
```

Register listeners once at setup:

```python
book.register_quoter_listener(mm.on_fill)     # origin="market_maker" fills routed here
book.register_hft_listener(hft.on_hft_fill)   # origin="hft" fills routed here
```

`_fire_fill` also decrements `remaining_size` on partial fills before dispatching the event, so `mm_resting_orders[oid]["remaining_size"]` is always current by the time the listener runs.

### Matching rules (`try_clear`)

- **Client-only cross**: `try_clear` matches **client orders against MM/HFT orders**. Client-vs-client and MM-vs-MM (incl. MM-vs-HFT) matches never happen.
- **Price-time priority**: best price first; `_seq` breaks ties (FIFO).
- **Crossing condition**: `client_buy.price ≥ mm_ask.price`, `client_sell.price ≤ mm_bid.price`.
- **Fill price**: always the passive (resting MM/HFT) price.
- **Partial fills**: remaining client size continues to walk the book; remaining MM/HFT size stays in the book and a `FillEvent(is_full_fill=False)` is fired.
- Sorted MM/HFT caches are rebuilt only when `_mm_dirty` is set (after `add_order` / `cancel_orders` on a maker-side order), so most `try_clear` calls skip the sort.

### Public API

| Method | Purpose |
|---|---|
| `tick(step)` | Record current simulation step (used for fill event timestamps and order ages) |
| `add_order(order)` | Add a single order; routes to `_orders` or `_client_orders` by `origin` |
| `post_mm_quotes(quotes)` | Batch-add MM quotes (convenience wrapper over `add_order`) |
| `route_client_order(order)` | `add_order` + immediate `try_clear` (single-order fast path) |
| `try_clear()` | Match all resting client orders against MM/HFT orders |
| `cancel_orders(ids)` | Remove specific MM/HFT order IDs |
| `cancel_all_mm_orders()` | Clear `_orders` + `_mm_resting` wholesale (legacy; Quoter uses specific `cancel_ids` instead) |
| `register_quoter_listener(cb)` | Subscribe to MM fills |
| `register_hft_listener(cb)` | Subscribe to HFT fills |
| `mm_resting_orders` *(property)* | Live dict view into `_mm_resting` — callers must not mutate |
| `display_mm_quotes()` | Pretty-print the MM ladder to stdout |
| `order_history` *(property)* | DataFrame of every submission when `track_submissions=True` |

---

## Typical usage

Inside a `Controller`-driven simulation ([`src/utils/report/`](../report/)), the book is driven once per step:

```python
book = Order_book()
book.register_quoter_listener(mm.on_fill)

for step in range(n_steps):
    book.tick(step)
    quotes, cancels = mm.compute_quotes(step, t, book.mm_resting_orders)
    book.cancel_orders(cancels)
    book.post_mm_quotes(quotes)
    for client_order in flow.generate_step(...):
        book.add_order(client_order)
    book.try_clear()                 # fires FillEvents to the registered listener
    mm.execute_hedge(step, t)
```

See [test/order_book.ipynb](../../../test/order_book.ipynb) for a standalone demo of the matching engine.
