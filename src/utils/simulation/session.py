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

import json
import pickle
import pathlib
import time
from typing import Callable, Any, Dict, List, Optional, Tuple


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


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _meta_path(pkl_path: pathlib.Path) -> pathlib.Path:
    """Return the companion .meta.json path for a given pkl path."""
    return pkl_path.with_suffix(".meta.json")


def _save_meta(pkl_path: pathlib.Path, meta: Dict) -> None:
    mp = _meta_path(pkl_path)
    with open(mp, "w") as f:
        json.dump(meta, f)


def _load_meta(pkl_path: pathlib.Path) -> Optional[Dict]:
    mp = _meta_path(pkl_path)
    if not mp.exists():
        return None
    with open(mp) as f:
        return json.load(f)


def _meta_matches(pkl_path: pathlib.Path, expected: Dict) -> bool:
    """
    Return True if every key in `expected` matches the stored meta.
    Returns False (stale) if meta file is missing or any value differs.
    Numeric values are compared with a small relative tolerance (1e-9).
    """
    stored = _load_meta(pkl_path)
    if stored is None:
        return False
    for k, v in expected.items():
        sv = stored.get(k)
        if sv is None:
            return False
        # Numeric comparison with tolerance
        try:
            if abs(float(sv) - float(v)) > 1e-9 * max(1.0, abs(float(v))):
                return False
        except (TypeError, ValueError):
            if sv != v:
                return False
    return True


# ── Cache helpers ──────────────────────────────────────────────────────────────

def cache(path, force: bool, fn: Callable,
          meta: Optional[Dict] = None) -> Any:
    """
    If force=False and path exists (and meta matches if provided): load and return.
    Otherwise: call fn(), save result to path, return result.

    meta : optional dict of key/value pairs saved alongside the pkl.
           On subsequent loads the stored values are compared against `meta`;
           a mismatch (e.g. dt changed) is treated as a cache miss and the
           object is recomputed automatically.

    Example
    -------
    stock = cache(path, force=False, fn=lambda: build(), meta={'dt': 0.1, 'n_days': 2})
    """
    path = pathlib.Path(path)
    stale = meta is not None and path.exists() and not _meta_matches(path, meta)
    if stale:
        print(f"[cache] STALE   {path.name}  (meta mismatch {meta}) — recomputing")
    if not force and not stale and path.exists():
        return load_obj(path)
    result = fn()
    save_obj(result, path)
    if meta is not None:
        _save_meta(path, meta)
    return result


def cache_group(paths: List, force: bool, fn: Callable,
                meta: Optional[Dict] = None) -> Tuple:
    """
    Like cache() but for a function that returns a tuple of objects,
    each saved to its own file.

    paths : list of file paths, one per returned object
    fn    : callable returning a tuple of the same length as paths
    meta  : optional dict — saved alongside every file and validated on load.
            Any mismatch (e.g. dt changed) invalidates the whole group.

    Example
    -------
    stock, mkt_B, mkt_C = cache_group(
        [dir/'stock.pkl', dir/'mkt_B.pkl', dir/'mkt_C.pkl'],
        force=False,
        fn=lambda: build_markets(seed=42, n_days=30, dt_seconds=0.05),
        meta={'dt': 0.05, 'n_days': 30, 'seed': 42},
    )
    """
    paths = [pathlib.Path(p) for p in paths]
    # Check staleness: any file missing or meta mismatch → recompute all
    all_exist = all(p.exists() for p in paths)
    if meta is not None and all_exist:
        stale = not all(_meta_matches(p, meta) for p in paths)
        if stale:
            print(f"[cache] STALE   {[p.name for p in paths]}  "
                  f"(meta mismatch {meta}) — recomputing")
    else:
        stale = False
    if not force and not stale and all_exist:
        return tuple(load_obj(p) for p in paths)
    results = fn()
    for obj, p in zip(results, paths):
        save_obj(obj, p)
        if meta is not None:
            _save_meta(p, meta)
    return tuple(results)


# ── Build factories ────────────────────────────────────────────────────────────

def build_markets(seed: int, n_days: int, dt_seconds: float,
                  alpha: float = 0.05, beta: float = 0.94,
                  lam: int = 100, sigma_J: float = 0.005,
                  drift: float = 0.0, vol: float = 0.07,
                  origin: float = 1.10,
                  _ref_dt: float = 0.05):
    """
    Simulate a GARCH price path and build reference markets B and C.

    GARCH alpha/beta are per-step parameters: the same (alpha, beta) values give
    different vol-cluster durations at different dt.  We rescale them so the
    vol-cluster half-life in *wall-clock seconds* stays constant regardless of dt.

    The reference calibration is _ref_dt=0.05 s.  At any other dt the persistence
    p = alpha+beta is mapped to the equivalent per-step persistence via:
        p_new = p_ref^(dt/ref_dt)
    and alpha/beta are kept proportional.

    Returns
    -------
    (stock, market_B, market_C)
    """
    import numpy as np
    from ..stock_simulation import Stock
    from ..report.fast_config import build_markets_B_C

    # ── Scale GARCH persistence for dt ────────────────────────────────────────
    if abs(dt_seconds - _ref_dt) > 1e-9:
        p_ref = alpha + beta
        # Clip persistence below 1 so the GARCH is stationary
        p_ref = min(p_ref, 0.9999)
        # Each step at dt_seconds must carry the same per-second persistence
        p_new = p_ref ** (dt_seconds / _ref_dt)
        ratio = (p_new / p_ref) if p_ref > 0 else 1.0
        alpha = alpha * ratio
        beta  = beta  * ratio

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

    book = Order_book()   # track_submissions=False (default) — avoids ~11M dict allocs
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
    # Smart sampling: always keep every step where a hedge fired, then fill the
    # remainder with uniform downsampling up to ~2000 total points.
    n_sl = len(sl)
    stride = max(1, n_sl // 2000)
    uniform_idx = set(range(0, n_sl, stride))
    if 'hedge_fired' in sl.columns:
        hedge_idx = set(sl.index[sl['hedge_fired'] > 0].tolist())
    else:
        hedge_idx = set()
    sample_idx = sorted(uniform_idx | hedge_idx)
    return {
        "fill_history":      mm._fill_history,
        "inventory":         mm.inventory,
        "step_log_sample":   sl.iloc[sample_idx].to_dict("records"),
        "n_quotes_posted":   ctrl._n_quotes_posted,
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

    mm._fill_history       = state["fill_history"]
    mm.inventory           = state["inventory"]
    ctrl._step_log         = state["step_log_sample"]
    ctrl._n_quotes_posted  = state.get("n_quotes_posted", 0)

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
