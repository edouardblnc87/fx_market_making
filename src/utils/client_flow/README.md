# `client_flow` — Poisson Client Order Flow

Models the arrival of client market and limit orders on Exchange A, grounded in Avellaneda & Stoikov (2006) §§2.4–2.5.

---

## Theory

Client orders arrive as a Poisson process with intensity

```
λ(δ) = A · exp(−k · δ)       δ in basis points
```

- **δ** — distance between the MM's best quote and the mid (bps, ≥ 0)
- **A** — baseline arrival rate at δ = 0 (per second)
- **k** — exponential decay per bp; half-life = ln(2) / k bps

Buy and sell sides are independent with separate `(A, k)` pairs.

Order sizes follow a **truncated Pareto**:

```
f(x) ∝ x^{−1−α}    x ∈ [size_min, size_max]
```

Sampling uses the inverse-CDF transform (see [size_model.py](size_model.py)).
Reference: Gabaix et al. (2003), Gopikrishnan et al. (2000).

---

## Files

| File | Role |
|---|---|
| [config.py](config.py) | Default parameters — `A_BUY/SELL`, `K_BUY/SELL`, `ALPHA`, `SIZE_MIN/MAX`, `MARKET_ORDER_RATIO` |
| [arrival.py](arrival.py) | `intensity(delta_bps, A, k)` and `sample_arrival_count(λ, dt, rng)` |
| [size_model.py](size_model.py) | `sample_size(alpha, x_min, x_max, rng) -> int` (rounded, floored at `x_min`) |
| [flow_generator.py](flow_generator.py) | `ClientFlowConfig`, `ClientFlowGenerator` |

> **Dead code:** [arrival.py:43](arrival.py#L43) defines `sample_arrival` (Bernoulli-style bool) which is not referenced anywhere in the codebase. Candidate for removal — `sample_arrival_count` covers all current uses.

---

## `ClientFlowGenerator`

```python
from utils.client_flow.flow_generator import ClientFlowGenerator, ClientFlowConfig

gen = ClientFlowGenerator(config=ClientFlowConfig(), seed=42)
```

### `generate_step(mid_price, best_bid, best_ask, dt) -> list[Order]`

Called once per simulation step (by `Controller`). Per side:

1. Compute `δ_bps = (mid − best_bid) / mid · 10_000` (bid side) and the symmetric ask-side distance.
2. `λ = intensity(δ_bps, A, k)`.
3. Draw `N ~ Poisson(λ · dt)` — **can exceed 1** per side, per step.
4. For each arrival, build an `Order` via `_build_order` (see below).

Returns a flat list of all orders (buy + sell) generated in this step.

### `generate_session(mid_prices, bid_prices, ask_prices, dt) -> list[tuple[int, Order]]`

Convenience wrapper: iterates `generate_step` over full arrays and returns `(step_index, Order)` pairs. Not used by `Controller` (which calls `generate_step` directly) — kept for notebook demos.

### Order construction (`_build_order`)

Each arrival draws a Pareto size, then flips market vs. limit with probability `market_order_ratio`:

- **Market order** — priced to cross immediately: buy at `best_ask`, sell at `best_bid`.
- **Limit order** — rests in the book at `mid ± (δ_bps / 10_000) · mid`, where `δ_bps ~ Exponential(1/k)`.

Price is rounded to 4 decimals (tick convention). `origin="client"` so the matching engine knows it can be crossed against MM orders.

---

## Parameters

Defaults in [config.py](config.py) (recalibrated April 2026):

| Parameter | Default | Meaning |
|---|---|---|
| `A_BUY` / `A_SELL` | `0.007` | Arrival rate at δ=0 (per second) |
| `K_BUY` / `K_SELL` | `0.3` | Decay per bp — half-life ≈ 2.31 bp |
| `ALPHA` | `1.5` | Pareto tail exponent |
| `SIZE_MIN` | `1 000` EUR | Truncation floor |
| `SIZE_MAX` | `100 000` EUR | Truncation ceiling |
| `MARKET_ORDER_RATIO` | `0.5` | Fraction of arrivals that are market orders |

See [test/client_flow.ipynb](../../../test/client_flow.ipynb) for a demonstration.
