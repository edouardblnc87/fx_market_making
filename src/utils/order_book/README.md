# order_book

Limit order book simulation with random order generation, matching engine, and cancellation.

## Files

| File | Role |
|------|------|
| `order_book_impl.py` | Core classes `Order` and `Order_book` |
| `graphic_utils.py` | `plot_order_book()` — cumulative depth chart |
| `config.py` | Constants: `ORDER_DIRECTION`, `ORDER_TYPE` |
| `utils.py` | Misc helpers |

---

## `order_book_impl.py`

### `Order`
Immutable value object representing a single order.

| Field | Description |
|-------|-------------|
| `id` | Unique ID — format `{timestamp_ms}_{4-digit random}` |
| `direction` | `"buy"` or `"sell"` |
| `price` | Float |
| `size` | Integer (drawn from normal distribution) |
| `type` | `"limit_order"` or `"market_order"` |
| `origin` | `"market_maker"` or `"client"` |
| `time` | Creation timestamp |

---

### `Order_book`
DataFrame-backed order book. Bids and asks are stored in a single DataFrame and exposed as filtered views.

**Key properties**
- `_df_order_book` — all orders
- `_df_bid_book` — buy orders only
- `_df_ask_book` — sell orders only
- `_df_matches` — history of matched trades

**Order generation**

Random orders are generated using truncated normal distributions anchored on the last seen price of each side. Two constraints are enforced:
Market maker orders only:
- Buy prices are capped at `best_ask - 0.001` — bids never cross the ask
- Sell prices are floored at `best_bid + 0.001` — asks never cross the bid

**Methods**

| Method | Description |
|--------|-------------|
| `add_order(order)` | Add a single `Order` to the book |
| `add_orders_batch(orders)` | Add a list of `Order` objects to the book |
| `try_clear()` | Match client orders against MM orders (price-time priority, partial fills). Records matches in `_df_matches`, removes fully filled orders |
| `cancel_orders(ids)` | Remove specific orders by ID list |
| `cancel_all_mm_orders()` | Remove all market maker orders (used before repricing) |

**Matching rules**
- Client orders only match against market maker orders (never client vs client)
- Client buy matches the cheapest MM ask where `client_price >= mm_ask_price`
- Client sell matches the highest MM bid where `client_price <= mm_bid_price`
- Partial fills supported: remaining size stays in the book, fully filled orders are removed
- Fill price = MM order price (passive side)

---

## Typical workflow

```python
ob = Order_book()

# 1. Market maker seeds the book
ob.add_orders_batch(mm_orders)

# 2. Client batch arrives
ob.add_orders_batch(client_orders)

# 3. Run clearing — matches crossing orders
ob.try_clear()

# 4. Inspect matches
print(ob._df_matches)

# 5. Market maker reprices
ob.cancel_all_mm_orders()
ob.add_orders_batch(new_mm_orders)
```
