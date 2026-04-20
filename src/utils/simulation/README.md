# `simulation` — Session Persistence

Pickle helpers for avoiding regeneration of expensive simulation objects between notebook runs. Market generation (GARCH paths, spread building, depth) takes tens of seconds — these helpers serialise it once and reload in milliseconds.

---

## Files

| File | Role |
|---|---|
| [session.py](session.py) | Public helpers — session and market (un)pickling |
| `saved_markets.pkl` | Default cache path for `save_markets` / `load_markets` |

---

## Public API

All four helpers are re-exported from [`utils.simulation`](__init__.py).

### `save_markets(stock, market_B, market_C, path=None)` / `load_markets(path=None)`

Cache only the three market objects (`Stock` + `Market B/C`). **Preferred** for notebook workflows: cheaper than a full session and sufficient because everything downstream (`Order_book`, `Quoter`, `Controller`) is cheap to rebuild.

```python
from utils.simulation import save_markets, load_markets

save_markets(stock, market_B, market_C)          # → saved_markets.pkl
stock, market_B, market_C = load_markets()        # ← saved_markets.pkl
```

Default path: `src/utils/simulation/saved_markets.pkl`. Override with `path=`.

### `save_session(stock, market_B, market_C, market_maker, book, controller, path=None)` / `load_session(path=None)`

Pickle a **full backtest** including `Quoter`, `Order_book`, and `Controller`. Use when you want to reload the exact post-simulation state (fills, inventory, `step_log`) without re-running.

```python
from utils.simulation import save_session, load_session

save_session(stock, market_B, market_C, mm, book, ctrl)
stock, market_B, market_C, mm, book, ctrl = load_session()
```

Default path: `src/utils/simulation/saved_session.pkl`.

---

## When to use which

| Scenario | Helper |
|---|---|
| Iterating on `Quoter` / `Controller` logic | `save_markets` — reload markets, rebuild everything else |
| Keeping a completed run for later analysis / plotting | `save_session` |

Both use `pickle.HIGHEST_PROTOCOL`. Files are git-ignored (`*.pkl` in `.gitignore`).
