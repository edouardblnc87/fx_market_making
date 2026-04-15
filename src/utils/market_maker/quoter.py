from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional

from ..order_book.order_book_impl import Order, Order_book
from ..order_book.events import FillEvent
from ..market_simulator.market import Market
from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR

# FX session boundaries in seconds from midnight UTC
# Tokyo: 00:00-08:00, London: 08:00-16:00, New York: 13:00-21:00
# We use 3 non-overlapping resets: 00:00, 08:00, 16:00 UTC
FX_SESSION_RESETS_UTC = [0, 8 * 3600, 16 * 3600]


@dataclass
class QuoterConfig:
    # ── Avellaneda-Stoikov core
    gamma: float = 0.1
    k:     float = 1.5
    T:     float = 8 * 3600

    # ── Guéant core
    alpha_spread: float = 0.5
    use_asymmetric_delta: bool = True
    terminal_penalty_strength: float = 5.0

    # ── Quoting structure
    n_levels: int   = 10
    beta:     float = 0.3
    Q_base:   float = 100_000.0

    # ── Market conventions
    tick_size: float = 0.0001

    # ── Requote triggers
    # Fraction of current spread: skip requote if price moved less than this
    requote_threshold_spread_fraction: float = 0.25
    # Staleness: force requote if order is older than this many steps AND inventory is stressed
    stale_steps: int = 300
    # Inventory stress threshold: fraction of capital_K above which staleness rule fires
    stale_inventory_fraction: float = 0.5

    # ── Volume weights
    weight_B: float = 0.75
    weight_C: float = 0.25

    # ── Latency parameters (seconds)
    latency_B_s:   float = 0.200
    latency_C_s:   float = 0.170
    latency_hft_s: float = 0.050

    # ── Volatility estimation
    vol_window: int = 6000

    # ── Order flow imbalance signal (Cartea & Jaimungal)
    # Tilt the reservation price toward the direction of dominant client flow.
    # imbalance_window: number of recent fills used to compute the signal
    # alpha_imbalance : scaling coefficient — how aggressively to tilt
    imbalance_window: int   = 50
    alpha_imbalance:  float = 0.0002

    # ── Risk management
    delta_limit: float = 0.90

    # ── Fee structure
    fee_A_maker: float = 0.0001
    fee_A_taker: float = 0.0004
    fee_B_maker: float = 0.00009
    fee_B_taker: float = 0.0002
    fee_C_maker: float = 0.00009
    fee_C_taker: float = 0.0003


@dataclass
class Quote:
    direction: str
    price:     float
    size:      float
    level:     int


class Quoter:
    """
    Phase 1 market maker for Exchange A.

    Producer side of the producer/listener pattern:
      - Emits quote updates each step (but only acts on the book selectively)
      - Listens for FillEvents from the OrderBook via on_fill()
      - Triggers session resets at FX session boundaries
    """

    def __init__(self, market_B:  Market, market_C:  Market, config:    QuoterConfig | None = None, capital_K: float = 1_000_000.0):
        self.market_B  = market_B
        self.market_C  = market_C
        self.cfg       = config if config is not None else QuoterConfig()
        self.capital_K = capital_K

        self.inventory: float = 0.0

        # Pending fills received from the OrderBook since last compute_quotes() call
        self._pending_fills: List[FillEvent] = []

        # Fill log for diagnostics
        self._fill_history: List[dict] = []

        # Previous theoretical quotes — used for threshold comparison
        self._prev_best_bid: float | None = None
        self._prev_best_ask: float | None = None

        # Last session reset timestamp (in simulation seconds)
        self._last_session_reset_t: float = -1.0

        # Order flow imbalance — rolling list of MM fill directions (capped at imbalance_window).
        # "sell" fill = client hit our ask -> bullish signal -> tilt reservation price up.
        # "buy"  fill = client hit our bid -> bearish signal -> tilt reservation price down.
        self._flow_history: List[str] = []

        dt = market_B.stock.time_step
        self._lag_B: int = max(1, round(self.cfg.latency_B_s / dt))
        self._lag_C: int = max(1, round(self.cfg.latency_C_s / dt))

        gap_B = self.cfg.latency_B_s - self.cfg.latency_hft_s
        gap_C = self.cfg.latency_C_s - self.cfg.latency_hft_s
        self._effective_gap_s: float = (self.cfg.weight_B * gap_B + self.cfg.weight_C * gap_C)

    
    #  LISTENER INTERFACE  (called by the OrderBook)

    def on_fill(self, event: FillEvent) -> None:
        """
        Callback registered with the OrderBook.
        Always updates inventory and logs the fill.
        Only queues a forced cancel/requote for full fills — a partially filled order
        is still resting and valid, so we leave it alone until threshold/staleness fires.
        """
        # Inventory update always applies regardless of partial/full
        delta_q = event.size if event.direction == "buy" else -event.size
        self.inventory += delta_q
        self._fill_history.append({
            "step":         event.step,
            "direction":    event.direction,
            "price":        event.price,
            "size":         event.size,
            "is_full_fill": event.is_full_fill,
        })

        # Track flow direction for the imbalance signal (rolling window)
        self._flow_history.append(event.direction)
        if len(self._flow_history) > self.cfg.imbalance_window:
            self._flow_history.pop(0)

        # Only force a requote if the order is fully consumed
        if event.is_full_fill:
            self._pending_fills.append(event)


    
    #  MAIN QUOTING FUNCTION

    def compute_quotes(self, step: int, t: float, resting_orders: dict) -> Tuple[List[Quote], List[str]]:
        """
        Compute a fresh 10-level bid/ask ladder and decide which resting orders to cancel.

        Returns
        -------
        new_quotes : Quote objects to submit. Empty if no requote needed.
        cancel_ids : IDs of resting MM orders to cancel before submitting.
        """

        # Collect IDs of fully-filled orders — these must be replaced even if price hasn't moved.
        # Partial fills are NOT in _pending_fills, so their orders stay untouched here.
        filled_ids  = {f.order_id  for f in self._pending_fills}
        filled_sides = {f.direction for f in self._pending_fills}
        self._pending_fills.clear()

        # ── Compute theoretical quotes
        bid_B, ask_B = self._get_stale_quotes(self.market_B, step, self._lag_B)
        bid_C, ask_C = self._get_stale_quotes(self.market_C, step, self._lag_C)
        best_bid_ref = max(bid_B, bid_C)
        best_ask_ref = min(ask_B, ask_C)
        fair_mid     = (best_bid_ref + best_ask_ref) / 2.0

        sigma            = self._estimate_vol(step)
        time_remaining_y = max(self.cfg.T - t, 1.0) / TRADING_SECONDS_PER_YEAR
        time_fraction    = t / self.cfg.T
        penalty_factor   = 1.0 + self.cfg.terminal_penalty_strength * (time_fraction ** 3)

        reservation_price = (
            fair_mid
            - self.inventory * self.cfg.gamma * sigma**2 * time_remaining_y * penalty_factor
        )

        # ── Order flow imbalance tilt (Cartea & Jaimungal)
        # imbalance in [-1, 1]: +1 = all client buys (hitting ask), -1 = all client sells (hitting bid).
        # When clients buy from us (hit ask), price is likely drifting up -> tilt reservation up.
        # When clients sell to us (hit bid), price is likely drifting down -> tilt reservation down.
        imbalance = self._compute_imbalance()
        reservation_price += self.cfg.alpha_imbalance * imbalance * fair_mid

        spread_AS_bps = (
            self.cfg.gamma * (sigma * 100.0) ** 2 * time_remaining_y
            + (2.0 / self.cfg.gamma) * np.log(1.0 + self.cfg.gamma / self.cfg.k)
        )
        spread_AS = spread_AS_bps / 10_000.0 * fair_mid

        sigma_per_s    = sigma / np.sqrt(TRADING_SECONDS_PER_YEAR)
        spread_latency = 2.0 * sigma_per_s * np.sqrt(self._effective_gap_s)

        inventory_ratio  = self.inventory / self.capital_K
        spread_inventory = self.cfg.alpha_spread * inventory_ratio**2 * spread_AS

        total_spread = spread_AS + spread_latency + spread_inventory
        half_spread  = max(total_spread / 2.0, self.cfg.tick_size)

        if self.cfg.use_asymmetric_delta:
            skew_delta = self.inventory * np.sqrt(
                self.cfg.gamma * sigma**2 / (2.0 * self.cfg.k)
            )
            max_skew   = half_spread - 0.5 * self.cfg.tick_size
            skew_delta = np.clip(skew_delta, -max_skew, max_skew)
        else:
            skew_delta = 0.0

        ask_half = half_spread - skew_delta
        bid_half = half_spread + skew_delta

        best_bid = self._snap_to_tick(reservation_price - bid_half)
        best_ask = self._snap_to_tick(reservation_price + ask_half)

        if best_bid >= best_ask:
            best_ask = best_bid + self.cfg.tick_size

        # ── Session reset check
        # At each FX session boundary: cancel everything, start clean.
        if self._is_session_reset(t):
            self._last_session_reset_t = t
            self._prev_best_bid = best_bid
            self._prev_best_ask = best_ask
            self._flow_history.clear()   # reset imbalance signal at each session boundary
            cancel_ids = list(resting_orders.keys())
            return self._build_ladder(best_bid, best_ask, inventory_ratio), cancel_ids

        # ── Selective cancel logic
        # Threshold in price units = fraction of current total spread
        threshold = self.cfg.requote_threshold_spread_fraction * total_spread

        inventory_stressed = abs(inventory_ratio) > self.cfg.stale_inventory_fraction

        cancel_ids: List[str] = []

        for oid, info in resting_orders.items():
            # Rule 1: filled orders must be replaced (already removed from book, but we track their side to ensure we requote it)
            if oid in filled_ids:
                cancel_ids.append(oid)
                continue

            resting_price = info["price"]
            side          = info["direction"]
            age           = info["age"]

            # New theoretical price for this side
            theo_price = best_bid if side == "buy" else best_ask

            # Rule 2: price has drifted beyond threshold
            if abs(resting_price - theo_price) > threshold:
                cancel_ids.append(oid)
                continue

            # Rule 3: order is stale AND inventory is stressed
            if age > self.cfg.stale_steps and inventory_stressed:
                cancel_ids.append(oid)

        # Only submit new quotes if there is something to cancel/replace OR a full fill happened
        if not cancel_ids and not filled_sides:
            if (self._prev_best_bid is not None
                    and abs(best_bid - self._prev_best_bid) <= threshold
                    and abs(best_ask - self._prev_best_ask) <= threshold):
                return [], []

        self._prev_best_bid = best_bid
        self._prev_best_ask = best_ask

        return self._build_ladder(best_bid, best_ask, inventory_ratio), cancel_ids

    
    
    #  RISK MANAGEMENT

    def needs_hedge(self) -> bool:
        return abs(self.inventory) / self.capital_K > self.cfg.delta_limit

    def hedge_order(self, market_B_depth: float, market_C_depth: float, fair_mid: float) -> Tuple[float, float, float]:
        total_size    = -self.inventory
        impact_factor = fair_mid * 0.0001

        score_B = (1.0 / (self.cfg.fee_B_taker + impact_factor / market_B_depth)
                   if market_B_depth > 0 else 0.0)
        score_C = (1.0 / (self.cfg.fee_C_taker + impact_factor / market_C_depth)
                   if market_C_depth > 0 else 0.0)

        total_score = score_B + score_C
        ratio_B     = self.cfg.weight_B if total_score == 0.0 else score_B / total_score

        size_B = round(total_size * ratio_B,         2)
        size_C = round(total_size * (1.0 - ratio_B), 2)

        fee_cost = (
            abs(size_B) * self.cfg.fee_B_taker * fair_mid
            + abs(size_C) * self.cfg.fee_C_taker * fair_mid
        )
        return size_B, size_C, fee_cost

    def fill_cost(self, size: float, fair_mid: float) -> float:
        return abs(size) * self.cfg.fee_A_maker * fair_mid

    

    #  DIAGNOSTICS

    def snapshot(self, step: int, t: float) -> dict:
        bid_B, ask_B = self._get_stale_quotes(self.market_B, step, self._lag_B)
        bid_C, ask_C = self._get_stale_quotes(self.market_C, step, self._lag_C)
        best_bid_ref  = max(bid_B, bid_C)
        best_ask_ref  = min(ask_B, ask_C)
        fair_mid      = (best_bid_ref + best_ask_ref) / 2.0

        sigma            = self._estimate_vol(step)
        time_remaining_y = max(self.cfg.T - t, 1.0) / TRADING_SECONDS_PER_YEAR
        time_fraction    = t / self.cfg.T
        penalty_factor   = 1.0 + self.cfg.terminal_penalty_strength * (time_fraction ** 3)

        reservation_price = (
            fair_mid
            - self.inventory * self.cfg.gamma * sigma**2 * time_remaining_y * penalty_factor
        )

        imbalance = self._compute_imbalance()
        reservation_price += self.cfg.alpha_imbalance * imbalance * fair_mid

        spread_AS_bps = (
            self.cfg.gamma * (sigma * 100.0) ** 2 * time_remaining_y
            + (2.0 / self.cfg.gamma) * np.log(1.0 + self.cfg.gamma / self.cfg.k)
        )
        spread_AS        = spread_AS_bps / 10_000.0 * fair_mid
        sigma_per_s      = sigma / np.sqrt(TRADING_SECONDS_PER_YEAR)
        spread_latency   = 2.0 * sigma_per_s * np.sqrt(self._effective_gap_s)
        inventory_ratio  = self.inventory / self.capital_K
        spread_inventory = self.cfg.alpha_spread * inventory_ratio**2 * spread_AS
        total_spread     = spread_AS + spread_latency + spread_inventory
        half_spread      = max(total_spread / 2.0, self.cfg.tick_size)

        if self.cfg.use_asymmetric_delta:
            skew_delta = self.inventory * np.sqrt(
                self.cfg.gamma * sigma**2 / (2.0 * self.cfg.k)
            )
            max_skew   = half_spread - 0.5 * self.cfg.tick_size
            skew_delta = np.clip(skew_delta, -max_skew, max_skew)
        else:
            skew_delta = 0.0

        ask_half = half_spread - skew_delta
        bid_half = half_spread + skew_delta
        best_bid = self._snap_to_tick(reservation_price - bid_half)
        best_ask = self._snap_to_tick(reservation_price + ask_half)

        return {
            "step":               step,
            "t_seconds":          t,
            "bid_B":              bid_B,  "ask_B": ask_B,
            "bid_C":              bid_C,  "ask_C": ask_C,
            "fair_mid":           fair_mid,
            "reservation_price":  reservation_price,
            "penalty_factor":     penalty_factor,
            "spread_AS":          spread_AS,
            "spread_latency":     spread_latency,
            "spread_inventory":   spread_inventory,
            "total_spread":       total_spread,
            "half_spread":        half_spread,
            "skew_delta":         skew_delta,
            "ask_half":           ask_half,
            "bid_half":           bid_half,
            "best_bid":           best_bid,
            "best_ask":           best_ask,
            "spread_bps":         (best_ask - best_bid) / fair_mid * 10_000.0,
            "inventory_EUR":      self.inventory,
            "inventory_ratio":    inventory_ratio,
            "time_remaining_s":   max(self.cfg.T - t, 1.0),
            "needs_hedge":        self.needs_hedge(),
            "effective_gap_ms":   self._effective_gap_s * 1000.0,
            "sigma_used":         sigma,
            "n_fills":            len(self._fill_history),
            "imbalance":          imbalance,
        }

    
    
    #  PRIVATE HELPERS

    def _build_ladder(self, best_bid: float, best_ask: float, inventory_ratio: float) -> List[Quote]:
        """Build the 10-level bid/ask ladder with inventory-skewed sizing."""
        inventory_skew = np.clip(inventory_ratio, -1.0, 1.0)
        quotes: List[Quote] = []
        for i in range(1, self.cfg.n_levels + 1):
            base_size = self.cfg.Q_base * np.exp(-self.cfg.beta * i)
            bid_size  = round(base_size * (1.0 - 0.5 * inventory_skew), 2)
            ask_size  = round(base_size * (1.0 + 0.5 * inventory_skew), 2)
            bid_price = self._snap_to_tick(best_bid - (i - 1) * self.cfg.tick_size)
            ask_price = self._snap_to_tick(best_ask + (i - 1) * self.cfg.tick_size)
            quotes.append(Quote("buy",  bid_price, bid_size, i))
            quotes.append(Quote("sell", ask_price, ask_size, i))
        return quotes

    def _is_session_reset(self, t: float) -> bool:
        """
        True if t has crossed a FX session boundary since the last reset.
        t is in seconds from simulation start (treated as seconds from midnight UTC).
        We check whether any reset boundary falls in (last_reset_t, t].
        """
        for boundary in FX_SESSION_RESETS_UTC:
            if self._last_session_reset_t < boundary <= t:
                return True
        return False

    def _estimate_vol(self, step: int) -> float:
        if step < self.cfg.vol_window:
            return self.market_B.stock.vol

        import pandas as pd
        start    = max(0, step - self.cfg.vol_window)
        prices   = self.market_B.noised_mid_price[start:step]
        log_rets = np.diff(np.log(prices))
        dt_frac  = self.market_B.stock.time_step / TRADING_SECONDS_PER_YEAR

        ewma_var     = pd.Series(log_rets).ewm(span=self.cfg.vol_window).var().iloc[-1]
        realized_vol = np.sqrt(ewma_var / dt_frac)
        return max(realized_vol, 0.2 * self.market_B.stock.vol)

    def _get_stale_quotes(self, market: Market, step: int, lag: int) -> Tuple[float, float]:
        stale_step = max(0, step - lag)
        return float(market.bid_price[stale_step]), float(market.ask_price[stale_step])

    def _compute_imbalance(self) -> float:
        """
        Order flow imbalance in [-1, 1] over the last imbalance_window fills.
        +1 = all client buys (hitting our ask) -> bullish signal.
        -1 = all client sells (hitting our bid) -> bearish signal.
        Returns 0 if no fill history yet.
        """
        if not self._flow_history:
            return 0.0
        n_ask_hits = self._flow_history.count("sell")  # client bought from us
        n_bid_hits = self._flow_history.count("buy")   # client sold to us
        total = n_ask_hits + n_bid_hits
        return (n_ask_hits - n_bid_hits) / total

    def _snap_to_tick(self, price: float) -> float:
        tick = self.cfg.tick_size
        return round(round(price / tick) * tick, 6)
