"""
session.py — Serialisation helpers and build/run factories for the backtest.

One pickle file per object convention:
    cache_dir/
        p1_stock.pkl
        p1_market_B.pkl
        p1_market_C.pkl
        p1_sim.pkl
        calibration.pkl
        p2_stock.pkl
        p2_market_B.pkl
        p2_market_C.pkl
        p2_sim.pkl

Public API
----------
cache(path, force, fn)            -- load or compute+save a single object
cache_group(paths, force, fn)     -- load or compute+save a tuple of objects
save_obj / load_obj               -- raw single-object pickle helpers

build_markets(seed, n_days, dt_seconds, **garch_kw)
    -> (stock, market_B, market_C)

run_sim(stock, market_B, market_C, quoter_config, capital, client_seed, n_steps)
    -> state dict  {fill_history, inventory, step_log_sample}

restore_ctrl(state, stock, market_B, market_C, quoter_config, capital, client_seed)
    -> (ctrl, mm, book)   controller shell with cached state restored
"""

from __future__ import annotations

import pickle
import pathlib
import time
from typing import Callable, Any, List, Tuple


# ── Raw pickle helpers ─────────────────────────────────────────────────────────

def save_obj(obj: Any, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] saved   {path.name}")


def load_obj(path) -> Any:
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[cache] loaded  {path.name}")
    return obj


# ── Cache helpers ──────────────────────────────────────────────────────────────

def cache(path, force: bool, fn: Callable) -> Any:
    """
    If force=False and path exists: load and return.
    Otherwise: call fn(), save result to path, return result.
    """
    path = pathlib.Path(path)
    if not force and path.exists():
        return load_obj(path)
    result = fn()
    save_obj(result, path)
    return result


def cache_group(paths: List, force: bool, fn: Callable) -> Tuple:
    """
    Like cache() but for a function that returns a tuple of objects,
    each saved to its own file.

    paths : list of file paths, one per returned object
    fn    : callable returning a tuple of the same length as paths

    Example
    -------
    stock, mkt_B, mkt_C = cache_group(
        [dir/'stock.pkl', dir/'mkt_B.pkl', dir/'mkt_C.pkl'],
        force=False,
        fn=lambda: build_markets(seed=42, n_days=30, dt_seconds=0.05),
    )
    """
    paths = [pathlib.Path(p) for p in paths]
    if not force and all(p.exists() for p in paths):
        return tuple(load_obj(p) for p in paths)
    results = fn()
    for obj, p in zip(results, paths):
        save_obj(obj, p)
    return tuple(results)


# ── Build factories ────────────────────────────────────────────────────────────

def build_markets(seed: int, n_days: int, dt_seconds: float,
                  alpha: float = 0.05, beta: float = 0.94,
                  lam: int = 100, sigma_J: float = 0.005,
                  drift: float = 0.0, vol: float = 0.07,
                  origin: float = 1.10):
    """
    Simulate a GARCH price path and build reference markets B and C.

    Returns
    -------
    (stock, market_B, market_C)
    """
    import numpy as np
    from ..stock_simulation import Stock
    from ..report.fast_config import build_markets_B_C

    np.random.seed(seed)
    stock = Stock(drift=drift, vol=vol, origin=origin)
    stock.simulate_garch(n_days=n_days, dt_seconds=dt_seconds,
                         alpha=alpha, beta=beta, lam=lam, sigma_J=sigma_J)
    market_B, market_C = build_markets_B_C(stock)
    return stock, market_B, market_C


def run_sim(stock, market_B, market_C, quoter_config,
            capital: float, client_seed: int,
            n_steps: int | None = None):
    """
    Build a fresh controller, run simulate(), and return a lightweight
    state dict suitable for pickling (no large numpy arrays).

    Parameters
    ----------
    n_steps : number of steps to simulate (default: all steps in stock)

    Returns
    -------
    dict with keys: fill_history, inventory, step_log_sample
    """
    from ..order_book.order_book_impl import Order_book
    from ..market_maker.quoter import Quoter
    from ..report.controller import Controller
    from ..client_flow.flow_generator import ClientFlowGenerator

    book = Order_book()
    mm   = Quoter(market_B, market_C, config=quoter_config, capital_K=capital)
    book.register_quoter_listener(mm.on_fill)

    gen = ClientFlowGenerator(seed=client_seed)
    ctrl = Controller(
        market_B, market_C, book, mm,
        lambda step, t, mid, bid, ask, dt:
            gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
    )

    n = n_steps if n_steps is not None else stock.n_steps
    t0 = time.time()
    ctrl.simulate(n)
    print(f"[sim] done in {time.time() - t0:.1f} s  ({n:,} steps)")

    sl = ctrl.step_log
    return {
        "fill_history":    mm._fill_history,
        "inventory":       mm.inventory,
        "step_log_sample": sl.iloc[::max(1, len(sl) // 2000)].to_dict("records"),
    }


def restore_ctrl(state: dict, market_B, market_C, quoter_config, capital: float,
                 client_seed: int):
    """
    Reconstruct a controller shell from a cached state dict.

    The controller is not re-simulated — it is populated with cached
    fill_history, inventory, and a downsampled step_log so that
    trade_history / pnl_report / report() all work.

    Returns
    -------
    (ctrl, mm, book)
    """
    from ..order_book.order_book_impl import Order_book
    from ..market_maker.quoter import Quoter
    from ..report.controller import Controller
    from ..client_flow.flow_generator import ClientFlowGenerator

    book = Order_book()
    mm   = Quoter(market_B, market_C, config=quoter_config, capital_K=capital)
    book.register_quoter_listener(mm.on_fill)

    gen = ClientFlowGenerator(seed=client_seed)
    ctrl = Controller(
        market_B, market_C, book, mm,
        lambda step, t, mid, bid, ask, dt:
            gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
    )

    mm._fill_history = state["fill_history"]
    mm.inventory     = state["inventory"]
    ctrl._step_log   = state["step_log_sample"]

    return ctrl, mm, book


# ── Legacy bundle helpers (kept for backwards compatibility) ───────────────────

_DEFAULT_PATH = pathlib.Path(__file__).parent / "saved_session.pkl"
_MARKETS_PATH = pathlib.Path(__file__).parent / "saved_markets.pkl"


def save_markets(stock, market_B, market_C, path=None):
    path = pathlib.Path(path) if path else _MARKETS_PATH
    with open(path, "wb") as f:
        pickle.dump({"stock": stock, "market_B": market_B, "market_C": market_C}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Markets saved → {path}")


def load_markets(path=None):
    path = pathlib.Path(path) if path else _MARKETS_PATH
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"Markets loaded ← {path}")
    return d["stock"], d["market_B"], d["market_C"]


def save_session(stock, market_B, market_C, market_maker, book, controller, path=None):
    path = pathlib.Path(path) if path else _DEFAULT_PATH
    session = {
        "stock": stock, "market_B": market_B, "market_C": market_C,
        "market_maker": market_maker, "book": book, "controller": controller,
    }
    with open(path, "wb") as f:
        pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Session saved → {path}")


def load_session(path=None):
    path = pathlib.Path(path) if path else _DEFAULT_PATH
    with open(path, "rb") as f:
        session = pickle.load(f)
    print(f"Session loaded ← {path}")
    return (session["stock"], session["market_B"], session["market_C"],
            session["market_maker"], session["book"], session["controller"])
