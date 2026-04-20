from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ..market_simulator import Market
from ..market_maker.quoter import Quoter, QuoterConfig
from ..order_book import Order_book
from ..client_flow.flow_generator import ClientFlowGenerator

from .controller import Controller

def _build_market_B(stock, seed=42):
    dt = stock.time_step
    np.random.seed(seed)
    m = Market(stock=stock)
    m.generate_noised_mid_price()
    # window_size at regime scale (120 s) — rolling RV is already smooth at
    # that scale, so no EMA smoothing is needed inside build_skewed_spread.
    # ema_span in wall-clock seconds → steps for the momentum signal.
    ema_span_steps = max(1, int(10 / dt))
    m.build_spread(option='Skew', window_size=max(1, int(60 / dt)), alpha=1.5,
                   gamma=0.3, ema_span=ema_span_steps, threshold=1.5,
                   spread_bps=0.5)   # 2×0.5×(1+1.5) = 2.5 bps at nominal vol, up to 5 bps at 5× vol
    m.generate_depth(mean_eur=500_000)
    return m

def _build_market_C(stock, seed=43):
    dt = stock.time_step
    np.random.seed(seed)
    m = Market(stock=stock)
    m.generate_noised_mid_price()
    m.build_spread(option='Adaptive', window_size=max(1, int(60 / dt)), alpha=0.8,
                   spread_bps=0.7)   # ~1.4 bps total at nominal vol (slightly wider than B)
    m.generate_depth(mean_eur=200_000)
    return m

def build_markets_B_C(stock):
    with ThreadPoolExecutor(max_workers=2) as pool:
        fb = pool.submit(_build_market_B, stock, seed=42)
        fc = pool.submit(_build_market_C, stock, seed=43)
        market_B = fb.result()
        market_C = fc.result()
    return market_B, market_C

def get_standard_demo_config():
    return QuoterConfig(
        gamma=0.01,                              # wider spread (~9 bps) → net edge above fees
        requote_threshold_spread_fraction=0.25,
        delta_limit=0.60,                        # hedge at 90% of each half-capital (EUR and USD)
        hedge_partial_limit=0.05,                # target < 5% of half-capital after hedge
        emergency_penalty_multiplier=5.0,
    )


CAPITAL = 1_000_000
def build_market_maker_and_order_book(market_1, market_2):
    book = Order_book()
    demo_mm = Quoter(market_1, market_2, config=get_standard_demo_config(), capital_K=CAPITAL)
    book.register_quoter_listener(demo_mm.on_fill)
    return demo_mm, book

class _ClientFlowFn:
    """Picklable wrapper around ClientFlowGenerator so sessions can be saved."""
    def __init__(self, seed):
        self._gen = ClientFlowGenerator(seed=seed)

    def __call__(self, step, t, mid, bid, ask, dt):
        return self._gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt)


def build_controller(market_1, market_2, book, marmet_maker, seed=44):
    client_flow_fn = _ClientFlowFn(seed=seed)
    return Controller(market_1, market_2, book, marmet_maker, client_flow_fn)
