from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from ..order_book.order_book_impl import Order, Order_book
from ..market_simulator.market import Market
from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR, TRADING_SECONDS_PER_DAY


@dataclass
class QuoterConfig:
    """All tunable parameters for the Phase 1 AS quoter."""

    # Avellaneda-Stoikov core
    gamma:        float = 0.1                     # risk-aversion coefficient
    k:            float = 1.5                     # order arrival decay: λ(δ) = A·exp(-k·δ)
    T:            float = TRADING_SECONDS_PER_DAY # session horizon (one trading day = 8 hours)

    # Quoting structure
    n_levels:     int   = 10        # price levels each side (project requirement)
    beta:         float = 0.3       # size decay: Q_i = Q_base · exp(-β·i)
    Q_base:       float = 100_000.0 # base EUR size at best level

    # Market conventions
    tick_size:    float = 0.0001    # 1bp tick, shared across all exchanges
    requote_threshold: int = 2      # only requote if best bid/ask moved more than N ticks

    # Volume weights (B=75%, C=25%)
    weight_B:     float = 0.75
    weight_C:     float = 0.25

    # Data latency in seconds (us=200ms/170ms, HFTs=50ms)
    latency_B_s:   float = 0.200
    latency_C_s:   float = 0.170
    latency_hft_s: float = 0.050

    # Adaptive volatility — rolling window
    # Falls back to parametric vol (stock.vol) when fewer than vol_window steps have been observed
    vol_window:   int   = 6000       # number of steps for realized vol estimation

    # Risk limit
    delta_limit:  float = 0.90     # hedge trigger: |q| / K > 90%

    # Fee structure (maker = limit orders, taker = market orders)
    fee_A_maker:  float = 0.0001
    fee_A_taker:  float = 0.0004
    fee_B_maker:  float = 0.00009
    fee_B_taker:  float = 0.0002   # preferred hedge venue
    fee_C_maker:  float = 0.00009
    fee_C_taker:  float = 0.0003   # fallback hedge venue


# QUOTE — one price level output by the quoter, to be translated into a limit order by the simulator
@dataclass
class Quote:
    direction: str   # "buy" or "sell"
    price:     float
    size:      float
    level:     int   # 1 = best, 10 = furthest


class Quoter:
    """
    Market maker for Exchange A — Avellaneda-Stoikov heuristic with latency premium.
    Designed to work across all phases: degrades gracefully when data is sparse (Phase 1)
    and improves automatically as history accumulates (Phase 2+).
    """

    def __init__(self, market_B: Market, market_C: Market, config: QuoterConfig | None = None, capital_K: float = 1_000_000.0):
        self.market_B  = market_B
        self.market_C  = market_C
        self.cfg       = config if config is not None else QuoterConfig()
        self.capital_K = capital_K

        self.inventory: float = 0.0
        self._live_order_ids: List[int] = []
        self._prev_best_bid: float | None = None

        # Fill history — accumulated across the session, used for vol estimation and k calibration in Phase 2
        self._fill_history: List[dict] = []

        # Pre-compute latency lags in number of time steps
        dt = market_B.stock.time_step
        self._lag_B: int = max(1, round(self.cfg.latency_B_s / dt))
        self._lag_C: int = max(1, round(self.cfg.latency_C_s / dt))

        # Effective latency gap vs HFTs, volume-weighted
        # B: 200ms - 50ms = 150ms behind | C: 170ms - 50ms = 120ms behind
        # effective_gap = 0.75 * 150ms + 0.25 * 120ms = 142.5ms
        gap_B = self.cfg.latency_B_s - self.cfg.latency_hft_s
        gap_C = self.cfg.latency_C_s - self.cfg.latency_hft_s
        self._effective_gap_s: float = self.cfg.weight_B * gap_B + self.cfg.weight_C * gap_C


    def compute_quotes(self, step: int, t: float) -> Tuple[List[Quote], List[int]]:
        """
        Main quoting function — called once per simulation step.

        Returns
        -------
        new_quotes : 20 Quote objects (10 bids + 10 asks)
        cancel_ids : IDs of previous orders to cancel before submitting new ones
        """

        # 1. Read stale bid/ask from B and C (latency-adjusted)
        bid_B, ask_B = self._get_stale_quotes(self.market_B, step, self._lag_B)
        bid_C, ask_C = self._get_stale_quotes(self.market_C, step, self._lag_C)

        # 2. Fair mid: best bid across venues, best ask across venues
        best_bid_ref = max(bid_B, bid_C)
        best_ask_ref = min(ask_B, ask_C)
        fair_mid = (best_bid_ref + best_ask_ref) / 2.0

        # 3. Adaptive volatility — use realized vol from rolling window if enough data,
        #    otherwise fall back to parametric vol from the Stock object
        sigma = self._estimate_vol(step)
        time_remaining_y = max(self.cfg.T - t, 1.0) / TRADING_SECONDS_PER_YEAR

        # 4. Reservation price: r(t) = fair_mid - q·γ·σ²·(T-t)
        reservation_price = fair_mid - self.inventory * self.cfg.gamma * sigma**2 * time_remaining_y

        # 5. Component A — AS spread (in bps then converted to price units)
        spread_AS_bps = (
            self.cfg.gamma * (sigma * 100) ** 2 * time_remaining_y
            + (2.0 / self.cfg.gamma) * np.log(1.0 + self.cfg.gamma / self.cfg.k)
        )
        spread_AS = spread_AS_bps / 10_000 * fair_mid

        # 6. Component B — latency premium
        sigma_per_s = sigma / np.sqrt(TRADING_SECONDS_PER_YEAR)
        spread_latency = 2.0 * sigma_per_s * np.sqrt(self._effective_gap_s)

        # 7. Total spread, floored at 1 tick
        total_spread = spread_AS + spread_latency
        half_spread  = max(total_spread / 2.0, self.cfg.tick_size)

        # 8. Best bid/ask on A, centered on reservation price
        best_bid = self._snap_to_tick(reservation_price - half_spread)
        best_ask = self._snap_to_tick(reservation_price + half_spread)
        if best_bid >= best_ask:
            best_ask = best_bid + self.cfg.tick_size

        # 9. Build 10 levels each side with skewed sizing based on inventory
        #    If long → more size on ask to attract buyers, less on bid
        #    If short → more size on bid to attract sellers, less on ask
        #    inventory_skew in [-1, 1], neutral when flat
        inventory_skew = np.clip(self.inventory / self.capital_K, -1.0, 1.0)
        new_quotes: List[Quote] = []
        for i in range(1, self.cfg.n_levels + 1):
            base_size = self.cfg.Q_base * np.exp(-self.cfg.beta * i)
            bid_size  = round(base_size * (1.0 - 0.5 * inventory_skew), 2)  # shrink bid when long
            ask_size  = round(base_size * (1.0 + 0.5 * inventory_skew), 2)  # grow ask when long
            new_quotes.append(Quote("buy",  self._snap_to_tick(best_bid - (i-1) * self.cfg.tick_size), bid_size, i))
            new_quotes.append(Quote("sell", self._snap_to_tick(best_ask + (i-1) * self.cfg.tick_size), ask_size, i))

        # 10. Only cancel-and-replace if quotes moved more than threshold
        prev_ids = list(self._live_order_ids)
        if self._prev_best_bid is not None:
            moved = abs(best_bid - self._prev_best_bid) / self.cfg.tick_size
            if moved < self.cfg.requote_threshold:
                return [], []

        self._prev_best_bid = best_bid
        self._live_order_ids = []

        return new_quotes, prev_ids


    def update_live_ids(self, ids: List[int]) -> None:
        """Register IDs of just-submitted orders so they can be cancelled next step."""
        self._live_order_ids = list(ids)

    def update_inventory(self, delta_q: float) -> None:
        """Update inventory after a fill. Positive = long EUR."""
        self.inventory += delta_q

    def record_fill(self, step: int, t: float, direction: str, price: float, size: float, delta: float) -> None:
        """
        Record a fill in the history. Called by the simulator on each execution.
        Accumulates data used for vol estimation and k calibration in Phase 2.
        delta = distance from fair_mid at time of fill (in price units).
        """
        self._fill_history.append({
            "step": step, "t": t,
            "direction": direction, "price": price,
            "size": size, "delta": delta,
        })

    def needs_hedge(self) -> bool:
        """True if |inventory| / K exceeds the delta risk limit."""
        return abs(self.inventory) / self.capital_K > self.cfg.delta_limit

    def hedge_order(self, market_B_depth: float, market_C_depth: float, fair_mid: float) -> Tuple[float, float, float]:
        """
        Compute hedge split across B and C, jointly optimising for depth and fees.

        The effective cost of hedging on a venue has two components:
          - Taker fee (fixed %)
          - Market impact: larger depth → lower impact, modelled as 1/depth

        We compute a score for each venue: score = 1 / (fee + impact_factor / depth)
        Higher score = cheaper venue. The hedge is then split proportionally to scores.

        Falls back to volume weights (75/25) if depth is zero on both venues.

        Returns (size_on_B, size_on_C, total_fee_cost), sizes signed (negative = sell EUR).
        """
        total_size = -self.inventory  # positive = need to buy EUR, negative = need to sell

        # Impact factor: how much 1 EUR of depth reduces market impact (tunable)
        impact_factor = fair_mid * 0.0001  # 1bp per unit of missing depth

        score_B = 1.0 / (self.cfg.fee_B_taker + (impact_factor / market_B_depth if market_B_depth > 0 else np.inf))
        score_C = 1.0 / (self.cfg.fee_C_taker + (impact_factor / market_C_depth if market_C_depth > 0 else np.inf))

        total_score = score_B + score_C
        if total_score == 0:
            ratio_B = self.cfg.weight_B  # fallback to volume weights
        else:
            ratio_B = score_B / total_score

        size_B = round(total_size * ratio_B, 2)
        size_C = round(total_size * (1.0 - ratio_B), 2)

        # Fee cost = taker fee on each venue × notional hedged
        fee_cost = abs(size_B) * self.cfg.fee_B_taker * fair_mid + abs(size_C) * self.cfg.fee_C_taker * fair_mid

        return size_B, size_C, fee_cost

    def fill_cost(self, size: float, fair_mid: float) -> float:
        """Maker fee paid on an Exchange A fill — deducted from realized P&L."""
        return abs(size) * self.cfg.fee_A_maker * fair_mid

    def snapshot(self, step: int, t: float) -> dict:
        """Return all intermediate quantities at current step — useful for backtesting report."""
        bid_B, ask_B = self._get_stale_quotes(self.market_B, step, self._lag_B)
        bid_C, ask_C = self._get_stale_quotes(self.market_C, step, self._lag_C)
        best_bid_ref  = max(bid_B, bid_C)
        best_ask_ref  = min(ask_B, ask_C)
        fair_mid      = (best_bid_ref + best_ask_ref) / 2.0
        sigma         = self._estimate_vol(step)
        time_remaining_y = max(self.cfg.T - t, 1.0) / TRADING_SECONDS_PER_YEAR
        reservation_price = fair_mid - self.inventory * self.cfg.gamma * sigma**2 * time_remaining_y
        spread_AS_bps = (self.cfg.gamma * (sigma * 100) ** 2 * time_remaining_y + (2.0 / self.cfg.gamma) * np.log(1.0 + self.cfg.gamma / self.cfg.k))
        spread_AS     = spread_AS_bps / 10_000 * fair_mid
        sigma_per_s   = sigma / np.sqrt(TRADING_SECONDS_PER_YEAR)
        spread_latency = 2.0 * sigma_per_s * np.sqrt(self._effective_gap_s)
        total_spread  = spread_AS + spread_latency
        half_spread   = max(total_spread / 2.0, self.cfg.tick_size)
        best_bid      = self._snap_to_tick(reservation_price - half_spread)
        best_ask      = self._snap_to_tick(reservation_price + half_spread)
        return {
            "step": step, "t_seconds": t,
            "bid_B": bid_B, "ask_B": ask_B,
            "bid_C": bid_C, "ask_C": ask_C,
            "fair_mid": fair_mid,
            "reservation_price": reservation_price,
            "spread_AS": spread_AS,
            "spread_latency": spread_latency,
            "total_spread": total_spread,
            "half_spread": half_spread,
            "best_bid": best_bid, "best_ask": best_ask,
            "spread_bps": (best_ask - best_bid) / fair_mid * 10_000,
            "inventory_EUR": self.inventory,
            "time_remaining_s": max(self.cfg.T - t, 1.0),
            "needs_hedge": self.needs_hedge(),
            "effective_gap_ms": self._effective_gap_s * 1000,
            "sigma_used": sigma,
            "n_fills": len(self._fill_history),
        }


    def _estimate_vol(self, step: int) -> float:
        if step < self.cfg.vol_window:
            return self.market_B.stock.vol

        start = max(0, step - self.cfg.vol_window)
        prices = self.market_B.noised_mid_price[start:step]
        log_rets = np.diff(np.log(prices))
        dt_frac = self.market_B.stock.time_step / TRADING_SECONDS_PER_YEAR

        # EWMA vol — more stable than rolling std at high frequency
        import pandas as pd
        ewma_vol = pd.Series(log_rets).ewm(span=self.cfg.vol_window).std().iloc[-1]
        realized_vol = ewma_vol / np.sqrt(dt_frac)

        return max(realized_vol, 0.2 * self.market_B.stock.vol)

    def _get_stale_quotes(self, market: Market, step: int, lag: int) -> Tuple[float, float]:
        """Read bid and ask from a market feed with latency lag applied."""
        stale_step = max(0, step - lag)
        return float(market.bid_price[stale_step]), float(market.ask_price[stale_step])

    def _snap_to_tick(self, price: float) -> float:
        """Round price to nearest tick."""
        tick = self.cfg.tick_size
        return round(round(price / tick) * tick, 6)
