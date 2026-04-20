from __future__ import annotations

import numpy as np
from typing import List

from ..order_book.order_book_impl import Order, Order_book, generate_order_id
from ..order_book.events import FillEvent
from ..market_simulator.market import Market
from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR
from .hft_config import HFTConfig
from .scenarios import HFTState, ScheduledEvent

TICK_SIZE = 0.0001


class HFTAgent:
    """
    HFT market maker on exchange A.

    Reprices every step using 1-step-lagged B/C data (50ms latency).
    Always quotes tighter than the MM (spread_fraction < 1).

    State machine:
        ACTIVE        — quotes both sides
        ONE_SIDED_BID — only bids (ask not competitive vs B/C)
        ONE_SIDED_ASK — only asks (bid not competitive vs B/C)
        OFFLINE       — no quotes

    State transitions (in priority order):
      1. Schedule overrides (forced windows from scenarios.py)
      2. Recovery timer: OFFLINE → ACTIVE after recovery_s
      3. Global: net A half-spread < min_net_half_spread_bps → OFFLINE
      4. Global: sigma > vol_offline_threshold → OFFLINE
      5. Otherwise → ACTIVE (both sides quoted)
      ONE_SIDED is exclusively schedule-driven.
    """

    def __init__(
        self,
        market_B: Market,
        market_C: Market,
        book: Order_book,
        config: HFTConfig,
        schedule: List[ScheduledEvent] | None = None,
        weight_B: float = 0.75,
        weight_C: float = 0.25,
    ) -> None:
        self.cfg        = config
        self._market_B  = market_B
        self._market_C  = market_C
        self._book      = book
        self._schedule  = schedule or []
        self._weight_B  = weight_B
        self._weight_C  = weight_C

        dt = market_B.stock.time_step
        self._dt          = dt
        self._lag         = max(1, round(config.latency_s / dt))
        self._trend_steps = max(1, round(config.trend_window_s / dt))

        self.state: HFTState = HFTState.ACTIVE
        self._inventory: float = 0.0
        self._offline_since_t: float | None = None   # tracks recovery timer

        # HFT's own order IDs — distinct name to never confuse with _mm_resting
        self._hft_resting_ids: set[str] = set()

        self._fill_history: list[dict] = []
        self._n_fills: int = 0

        # Precomputed A-S constants (same params as our quoter defaults)
        # HFT uses same model but tighter via spread_fraction
        self._gamma: float = 0.1
        self._omega: float = 1.0 / (8 * 3600)
        self._k:     float = 0.3
        self._inv_horizon_y: float = 1.0 / (self._omega * TRADING_SECONDS_PER_YEAR)
        self._two_over_gamma: float = 2.0 / self._gamma
        self._log_term: float = np.log(1.0 + self._gamma / self._k)  # constant: never changes
        self._sigma: float = market_B.stock.vol   # updated each step via EWMA

        # Lightweight EWMA variance for vol estimation
        self._ewma_var: float | None = None
        self._ewma_prev_price: float | None = None
        self._dt_frac: float = dt / TRADING_SECONDS_PER_YEAR
        self._inv_dt_frac: float = TRADING_SECONDS_PER_YEAR / dt  # avoids division in _update_vol
        self._ewma_alpha: float = 2.0 / (600 + 1)                 # constant EWMA span

        # Precomputed config-derived constants
        self._inv_tick: float = 1.0 / TICK_SIZE        # avoids division in _snap (called 2×/step)
        self._fee_bps: float = config.fee_A_maker * 1e4  # avoids multiplication in _update_state

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self, step: int, t: float) -> None:
        """Called once per simulation step, before client orders arrive."""
        # 1. Cancel all resting HFT orders
        self._book.cancel_orders(list(self._hft_resting_ids))
        self._hft_resting_ids.clear()

        # 2. Update vol estimate
        self._update_vol(step)

        # 3. Compute spread_AS_bps and stale index once — shared with _update_state
        sigma = self._sigma
        spread_AS_bps = (
            self._gamma * sigma * sigma * 1e4 * self._inv_horizon_y
            + self._two_over_gamma * self._log_term
        )
        stale = max(0, step - self._lag)
        self._update_state(t, step, stale, spread_AS_bps)

        if self.state == HFTState.OFFLINE:
            return

        # 4. Read B/C with HFT latency lag (stale already computed above)
        bid_B = float(self._market_B.bid_price[stale])
        ask_B = float(self._market_B.ask_price[stale])
        bid_C = float(self._market_C.bid_price[stale])
        ask_C = float(self._market_C.ask_price[stale])
        fair_mid = (
            self._weight_B * (bid_B + ask_B) * 0.5
            + self._weight_C * (bid_C + ask_C) * 0.5
        )

        # 5. Tight half-spread (A-S × spread_fraction; spread_AS_bps already computed)
        half_spread = max(
            spread_AS_bps / 10_000.0 * fair_mid * self.cfg.spread_fraction,
            TICK_SIZE,
        )

        # 6. Quote symmetrically
        best_bid = self._snap(fair_mid - half_spread)
        best_ask = self._snap(fair_mid + half_spread)
        if best_bid >= best_ask:
            best_ask = best_bid + TICK_SIZE

        # 7. Post orders based on state (set by _update_state)
        size = self.cfg.max_depth_eur
        if self.state in (HFTState.ACTIVE, HFTState.ONE_SIDED_BID):
            oid = generate_order_id()
            self._book.add_order(Order(oid, "buy", best_bid, size, "limit_order", "hft", 0))
            self._hft_resting_ids.add(oid)

        if self.state in (HFTState.ACTIVE, HFTState.ONE_SIDED_ASK):
            oid = generate_order_id()
            self._book.add_order(Order(oid, "sell", best_ask, size, "limit_order", "hft", 0))
            self._hft_resting_ids.add(oid)

    def on_hft_fill(self, event: FillEvent) -> None:
        """Registered on the order book for fills on HFT orders."""
        delta = event.size if event.direction == "buy" else -event.size
        self._inventory += delta
        self._n_fills += 1
        self._fill_history.append({
            "step":      event.step,
            "direction": event.direction,
            "price":     event.price,
            "size":      event.size,
        })

    @property
    def fill_history(self):
        import pandas as pd
        return pd.DataFrame(self._fill_history)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_state(self, t: float, step: int, stale_now: int, spread_AS_bps: float) -> None:
        # Schedule override takes priority
        day = t / 86_400.0
        for event in self._schedule:
            if event.start_day <= day < event.start_day + event.duration_days:
                self.state = event.state
                if self.state == HFTState.OFFLINE:
                    if self._offline_since_t is None:
                        self._offline_since_t = t
                return

        # Recovery from OFFLINE
        if self.state == HFTState.OFFLINE:
            if (self._offline_since_t is not None
                    and t - self._offline_since_t >= self.cfg.recovery_s):
                self.state = HFTState.ACTIVE
                self._offline_since_t = None
            return

        # ── Primary trigger: profitability check ─────────────────────────────
        if spread_AS_bps * self.cfg.spread_fraction - self._fee_bps < self.cfg.min_net_half_spread_bps:
            self.state = HFTState.OFFLINE
            self._offline_since_t = t
            return

        # ── Secondary trigger: extreme vol stress ────────────────────────────
        if self._sigma > self.cfg.vol_offline_threshold:
            self.state = HFTState.OFFLINE
            self._offline_since_t = t
            return

        # ── Directional trend → adverse selection avoidance ──────────────────
        stale_past = max(0, stale_now - self._trend_steps)
        try:
            curr_mid = (float(self._market_B.bid_price[stale_now])
                        + float(self._market_B.ask_price[stale_now])) * 0.5
            past_mid = (float(self._market_B.bid_price[stale_past])
                        + float(self._market_B.ask_price[stale_past])) * 0.5
            if past_mid > 0:
                move_bps = (curr_mid - past_mid) / past_mid * 1e4
                if move_bps > self.cfg.trend_threshold_bps:
                    self.state = HFTState.ONE_SIDED_BID
                    return
                if move_bps < -self.cfg.trend_threshold_bps:
                    self.state = HFTState.ONE_SIDED_ASK
                    return
        except (IndexError, AttributeError):
            pass

        self.state = HFTState.ACTIVE

    def _update_vol(self, step: int) -> None:
        try:
            p1 = float(self._market_B.noised_mid_price[step])
        except (IndexError, AttributeError):
            return
        if self._ewma_prev_price is None:
            self._ewma_prev_price = p1
            return
        p0 = self._ewma_prev_price
        self._ewma_prev_price = p1
        if p0 <= 0:
            return
        r = np.log(p1 / p0)
        v = r * r
        self._ewma_var = v if self._ewma_var is None else self._ewma_alpha * v + (1 - self._ewma_alpha) * self._ewma_var
        if self._ewma_var is not None:
            self._sigma = max(np.sqrt(self._ewma_var * self._inv_dt_frac), 1e-6)

    def _snap(self, price: float) -> float:
        return round(round(price * self._inv_tick) * TICK_SIZE, 6)
