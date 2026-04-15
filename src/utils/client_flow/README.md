# `client_flow` — Poisson Client Order Flow

Models the arrival of client market and limit orders on Exchange A, grounded in Avellaneda & Stoikov (2006) Sections 2.4–2.5.

---

## Theory

Client orders arrive as a Poisson process with intensity:

```
λ(δ) = A · exp(-k · δ)       δ in basis points
```

- **δ** — distance between the MM's quote and the mid-price (bps)
- **A** — baseline arrival rate at δ = 0 (per second)
- **k** — decay rate per bp; half-life = ln(2)/k bps

Buy and sell sides are independent with separate (A, k) parameters.

Order sizes follow a truncated Pareto (power law):

```
f(x) ∝ x^{-1-α}    x ∈ [size_min, size_max]
```

with α ≈ 1.5 (Gabaix et al. 2003, Gopikrishnan et al. 2000).

---

## Files

| File | Role |
|------|------|
| `config.py` | Default parameters — A, k, α, size bounds, MO ratio |
| `arrival.py` | `intensity(delta_bps, A, k)` and `sample_arrival(λ, dt, rng)` |
| `size_model.py` | `sample_size(alpha, x_min, x_max, rng)` — truncated Pareto via inverse CDF |
| `flow_generator.py` | `ClientFlowGenerator` — main class |

---

## Key class: `ClientFlowGenerator`

```python
from utils.client_flow.flow_generator import ClientFlowGenerator, ClientFlowConfig

cfg = ClientFlowConfig(A_buy=1.5, A_sell=1.5, k_buy=1.5, k_sell=1.5)
gen = ClientFlowGenerator(config=cfg, seed=42)
```

**`generate_step(mid_price, best_bid, best_ask, dt) -> list[Order]`**
Called once per simulation step. Returns 0, 1, or 2 `Order` objects (one per side max).

**`generate_session(mid_prices, bid_prices, ask_prices, dt) -> list[tuple[int, Order]]`**
Generates all client orders for a full session. Returns `(step_index, Order)` pairs.

---

## Order types

Each arrival is randomly assigned:
- **Market order** (50% default): priced to cross immediately — buy at `best_ask`, sell at `best_bid`
- **Limit order** (50% default): placed at `mid ± δ` where δ ~ Exponential(k) in bps

Both types use `origin="client"` and feed directly into `Order_book.add_orders_batch`.

---

## Integration with Exchange A

```python
# MM has already quoted on Exchange A
ob._generate_n_random_order(40)          # seed MM book

# Generate client flow and submit step by step
for step, order in gen.generate_session(mid, bid, ask, dt=0.1):
    ob.add_orders_batch([order])         # triggers _try_clear → matches vs MM quotes
```

---

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `A_buy` / `A_sell` | 1.5 | Arrival rate at δ=0 (per second) |
| `k_buy` / `k_sell` | 1.5 | Decay per bp — half-life ≈ 0.46 bp |
| `alpha` | 1.5 | Pareto tail exponent |
| `size_min` | 1 000 EUR | Minimum order size |
| `size_max` | 100 000 EUR | Maximum order size |
| `market_order_ratio` | 0.5 | Fraction of arrivals that are market orders |

See `test/client_flow.ipynb` for a full demonstration.
