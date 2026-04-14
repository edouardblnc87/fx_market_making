# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Market making / electronic trading simulation project (M2 course). Implements an order book simulation with random order generation and visualization.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Run main entry point
uv run main.py

# Launch Jupyter for the main notebook
uv run jupyter notebook src/order_book.ipynb

# Add a dependency
uv add <package>
```

There are no tests configured yet.

## Architecture

The core simulation lives in `src/utils/`:

- **`order_book_utils.py`** — central module. Contains two classes:
  - `Order`: immutable value object (id, direction, price, size, type, timestamp)
  - `Order_book`: DataFrame-backed limit order book. Separates bids/asks via filtered views (`_df_bid_book`, `_df_ask_book`). Generates synthetic orders using truncated normal distributions — buys are capped below 110% of last price, sells are floored above 90% of last price.
- **`graphic_utils.py`** — `plot_order_book()` renders a dark-themed cumulative depth chart using matplotlib step plots.
- **`config.py`** — constants: `ORDER_DIRECTION`, `ORDER_TYPE`.
- **`utils.py`** — misc helpers (currently just `generate_dataframe`).

The main interactive environment is the Jupyter notebook at `src/order_book.ipynb`. Imports use `importlib.reload()` for iterative development without restarting the kernel. The notebook's working directory is `src/`, so imports are relative (e.g. `import utils.order_book_utils`).

Reference books (market microstructure, HFT, liquidity) are in `books/`.

---

## Stock & Market simulation (`src/utils/stock_simulation/` and `src/utils/market_simulator/`)

### Stock (`stock_simulation/stock.py`)

`Stock` simulates a GBM price path. Key attributes after `simulate_gbm()`:

| Attribute | Description |
|-----------|-------------|
| `self.simulation` | 1D array shape `(N+1,)` — the price path |
| `self._time_grid` | 1D array shape `(N+1,)` — time in seconds |
| `self.time_step` | `dt` in seconds |
| `self.n_steps` | `N` (number of steps) |
| `self.vol_realized` | per-step log-return residuals `(N,)` |

**IMPORTANT**: the attribute is `self.simulation` (singular), not `self.simulations`.

Methods:
- `simulate_gbm(n_days, dt_seconds)` — generates the GBM path
- `plot_path()` — dark-themed price path plot (cyan `#00bfff`)
- `sanity_check()` — prints stats table (realized vol/drift, skewness, kurtosis, price range) + 2-subplot figure (price path + log-return histogram with fitted normal)

Constants used: `TRADING_SECONDS_PER_YEAR` from `stock_simulation/config.py`.

### Market (`market_simulator/market.py`)

`Market` wraps a `Stock` and adds i.i.d. microstructure noise to produce an observed mid price.

Methods:
- `generate_noise(vol_factor)` — draws `N+1` normals, stores in `self.vol_noise`
- `generate_noised_mid_price(vol_factor=0.1)` — sets `self.noised_mid_price = stock.simulation + vol_noise`
- `plot_noised_mid_price()` — dark-themed plot in orange `#ff9500`
- `plot_comparison()` — 2 stacked subplots: both paths (cyan vs orange) + deviation fill
- `sanity_check()` — prints noise std, noise vol, total vol, GBM vol, noise-to-signal ratio

**Import**: uses relative import `from ..stock_simulation import Stock` (no sys.path hacks).

`market_simulator/__init__.py` exports `Market`.

### Usage
```python
from utils.stock_simulation import Stock
from utils.market_simulator import Market

s = Stock(drift=0.0, vol=0.20)
s.simulate_gbm(n_days=1)
s.sanity_check()

m = Market(s)
m.generate_noised_mid_price(vol_factor=0.1)
m.plot_noised_mid_price()
m.plot_comparison()
m.sanity_check()
```
