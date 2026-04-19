from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Set

from ..order_book.order_book_impl import Order, Order_book, generate_order_id
from ..order_book.events import FillEvent
from ..market_simulator.market import Market
from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR

# FX session boundaries in seconds from midnight UTC.
# Tokyo: 00:00-08:00  |  London: 08:00-16:00  |  New York: 16:00-00:00
FX_SESSION_RESETS_UTC = [8 * 3600, 16 * 3600]

# Session-adaptive k values: order-arrival intensity decay varies significantly across FX sessions. 
# London/NY overlap is the deepest and most liquid; Tokyo is comparatively thin. A single k calibrated on one session misprices fill probability on others.
# These multipliers are applied to cfg.k at quote time via _session_k().
_SESSION_K_MULTIPLIERS = {
    "tokyo":   1.00, #0.75,   # thinner book, orders less sensitive to distance → lower k
    "london":  1.00, #1.25,   # deepest session, strong price sensitivity → higher k
    "newyork": 1.00,   # reference calibration
}


@dataclass
class QuoterConfig:
    # Avellaneda-Stoikov core
    gamma: float = 0.1
    k: float = 0.3
    # Infinite-horizon discount rate (replaces T-t from the finite-horizon A-S formulation).
    omega: float = 1.0 / (8 * 3600)   # default: risk horizon ≈ one FX session

    # Guéant core
    alpha_spread: float = 0.5
    use_asymmetric_delta: bool = True

    # Quoting structure
    n_levels: int = 10
    beta: float = 0.3
    Q_base: float = 100_000.0

    # Market conventions
    tick_size: float = 0.0001

    # Requote triggers
    requote_threshold_spread_fraction: float = 0.25
    stale_s: float = 3.0            # seconds before Priority 6 fires (was stale_steps)
    stale_tight_fraction: float = 0.88  # reprice stale orders at 70% of normal half_spread
    # Inventory-change trigger: force a requote when |inventory| has moved by more than this fraction of capital_K since the last reprice, even if prices are flat.
    inventory_requote_fraction: float = 0.05

    # Minimum number of fills required before the imbalance signal is trusted.
    imbalance_min_samples: int = 10  # Below this count the window is too small and a single trade flips the signal.

    # Volume weights
    weight_B: float = 0.75
    weight_C: float = 0.25

    # Latency parameters (seconds)
    latency_B_s: float = 0.200
    latency_C_s: float = 0.170
    latency_hft_s: float = 0.050

    # Volatility estimation
    vol_window_s: float = 60.0      # EWMA lookback in seconds (was vol_window steps)
    vol_min_s:    float = 2.0       # warm-up before switching to realized vol (was vol_min_steps)

    # Order flow imbalance signal (Cartea & Jaimungal)
    imbalance_window: int = 50
    alpha_imbalance: float = 0.0002

    # Risk management — fractions of half-capital (capital_K / 2).
    # Capital is split 50/50 between EUR and USD at inception, so limits are
    # expressed as a fraction of each half independently.
    # e.g. delta_limit=0.90 → hedge fires when |EUR inventory| > 0.90 × (K/2)
    #                        OR |EUR inv × price| (USD notional) > 0.90 × (K/2)
    delta_limit: float = 0.90
    hedge_partial_limit: float = 0.80
    emergency_penalty_multiplier: float = 5.0
    # EOD flat: force inventory to zero every eod_flat_interval seconds.
    # Set to 0.0 to disable (continuous running, no daily reset).
    eod_flat_interval: float = 0. #86_400.0

    # Fee structure
    fee_A_maker: float = 0.0001
    fee_A_taker: float = 0.0004
    fee_B_maker: float = 0.00009
    fee_B_taker: float = 0.0002
    fee_C_maker: float = 0.00009
    fee_C_taker: float = 0.0003


class Quoter:
    """
    Phase 1 market maker for Exchange A.

    Producer side of the producer/listener pattern:
      - Emits quote updates each step (but only acts on the book selectively)
      - Listens for FillEvents from the OrderBook via on_fill()
      - Triggers session resets at FX session boundaries

    Key modelling choices vs the original:
      - Infinite-horizon A-S formulation (omega replaces T-t): stationary inventory
        penalty appropriate for a 24/5 FX market with no terminal liquidation.
      - Session-adaptive k: order-arrival intensity scaled per FX session liquidity.
      - Inventory-change requote trigger: prevents silent accumulation at stale skew.
      - Hedge fill price: taker spread on B/C included in hedge P&L accounting.
      - Vol fallback: parametric vol used as-is below vol_min_steps (no 0.2× floor).
    """

    def __init__(self, market_B: Market, market_C: Market, config: QuoterConfig | None = None, capital_K: float = 1_000_000.0):
        self.market_B = market_B
        self.market_C = market_C
        self.cfg = config if config is not None else QuoterConfig()
        self.capital_K = capital_K

        self.inventory: float = 0.0

        # Pending fills received from the OrderBook since last compute_quotes() call
        self._pending_fills: List[FillEvent] = []

        # Partial fills pending a top-up (restore level to original size)
        self._pending_topups: List[FillEvent] = []

        # Fill log for diagnostics
        self._fill_history: List[dict] = []

        # Previous theoretical quotes — used for threshold comparison
        self._prev_best_bid: float | None = None
        self._prev_best_ask: float | None = None

        # Inventory level at the last reprice — used for the inventory-change trigger
        self._prev_inventory: float = 0.0

        # Flag: True until compute_quotes() has been called at least once.
        self._first_quote: bool = True

        # Last requote rule that fired: 0=nothing, 1-6=priority
        self._last_requote_rule: int = 0

        # True when a hedge was attempted but available depth on B+C wasn't enough to bring inventory below hedge_partial_limit. 
        self._hedge_emergency: bool = False  # Compute_quotes will apply an emergency penalty. Multiplier to force extreme quote skew on A until inventory recovers.

        # Last session reset timestamp (in simulation seconds).
        self._last_session_reset_t: float = 0.0

        # EOD flat tracking: last day-boundary we have already flattened.
        self._last_eod_day: int = -1

        # Order flow imbalance — rolling deque of MM fill directions.
        self._flow_history: deque = deque()
        self._imbalance_n_sells: int = 0
        self._imbalance_n_buys:  int = 0

        # Incremental EWMA variance for vol estimation (O(1) per step).
        self._ewma_var: float | None = None

        dt = market_B.stock.time_step
        self._dt: float = dt
        # Convert second-based config windows to step counts once at init
        self._vol_window_steps: int = max(1, round(self.cfg.vol_window_s / dt))
        self._vol_min_steps:    int = max(1, round(self.cfg.vol_min_s    / dt))
        self._stale_steps:      int = max(1, round(self.cfg.stale_s      / dt))
        self._lag_B: int = max(1, round(self.cfg.latency_B_s / dt))
        self._lag_C: int = max(1, round(self.cfg.latency_C_s / dt))

        gap_B = self.cfg.latency_B_s - self.cfg.latency_hft_s
        gap_C = self.cfg.latency_C_s - self.cfg.latency_hft_s
        self._effective_gap_s: float = (self.cfg.weight_B * gap_B + self.cfg.weight_C * gap_C)

        # ── Precomputed constants (never change after __init__) ───────────────
        self._inv_capital_K: float  = 1.0 / capital_K
        self._inv_horizon_y: float  = 1.0 / (self.cfg.omega * TRADING_SECONDS_PER_YEAR)
        self._two_over_gamma: float = 2.0 / self.cfg.gamma
        # latency spread: 2·σ_per_s·√gap  →  2·√(gap/TSPY)·σ  (sigma factored out)
        self._latency_coeff: float  = 2.0 * np.sqrt(self._effective_gap_s / TRADING_SECONDS_PER_YEAR)
        # dt fraction used in _estimate_vol
        self._dt_frac: float        = market_B.stock.time_step / TRADING_SECONDS_PER_YEAR

        # ── Per-step caches ───────────────────────────────────────────────────
        # Stale B/C quotes set by compute_quotes each step; reused by execute_hedge
        # on the rare steps when a hedge fires, avoiding a second array lookup.
        self._cache_bid_B: float = 0.0
        self._cache_ask_B: float = 0.0
        self._cache_bid_C: float = 0.0
        self._cache_ask_C: float = 0.0
        # Sigma set by compute_quotes; reused by execute_hedge (avoids second sqrt)
        self._cache_sigma: float = market_B.stock.vol
        # Previous mid-price for EWMA: avoids reading noised_mid_price[step-1]
        # every step (it was already read as p1 at the previous step).
        self._ewma_prev_price: float | None = None


    #  LISTENER INTERFACE  (called by the OrderBook)

    def on_fill(self, event: FillEvent) -> None:
        """
        Callback registered with the OrderBook.
        Always updates inventory and logs the fill.
        Only queues a forced cancel/requote for full fills — a partially filled order
        is still resting and valid, so we leave it alone until threshold/staleness fires.
        """
        delta_q = event.size if event.direction == "buy" else -event.size
        self.inventory += delta_q

        t_fill = event.step * self._dt
        mid_B = (float(self.market_B.bid_price[event.step]) + float(self.market_B.ask_price[event.step])) / 2.0
        mid_C = (float(self.market_C.bid_price[event.step]) + float(self.market_C.ask_price[event.step])) / 2.0
        fair_mid = self.cfg.weight_B * mid_B + self.cfg.weight_C * mid_C
        fee_cost = event.size * event.price * self.cfg.fee_A_maker

        # MM perspective: sell fill = cash in, buy fill = cash out — fees proportional to notional
        cash_flow = (event.size * event.price * (1.0 - self.cfg.fee_A_maker)
                     if event.direction == "sell"
                     else -event.size * event.price * (1.0 + self.cfg.fee_A_maker))

        self._fill_history.append({
            "order_id": event.order_id,
            "level": event.level,
            "step": event.step,
            "t": t_fill,
            "direction": event.direction,
            "price": event.price,
            "size": event.size,
            "is_full_fill": event.is_full_fill,
            "fair_mid": fair_mid,
            "fee_cost": fee_cost,
            "cash_flow": cash_flow,
            "inventory_after": self.inventory,
            "is_hedge": False,
            "is_eod_flat": False,
            "venue": "A",
        })

        self._flow_history.append(event.direction)
        if event.direction == "sell":
            self._imbalance_n_sells += 1
        else:
            self._imbalance_n_buys += 1
        while len(self._flow_history) > self.cfg.imbalance_window:
            removed = self._flow_history.popleft()
            if removed == "sell":
                self._imbalance_n_sells -= 1
            else:
                self._imbalance_n_buys -= 1

        if event.is_full_fill:
            self._pending_fills.append(event)
        else:
            self._pending_topups.append(event)


    #  MAIN QUOTING FUNCTION

    def compute_quotes(self, step: int, t: float, resting_orders: dict) -> Tuple[List[Order], List[str]]:
        """
        Compute the MM quote ladder and decide which resting orders to cancel/replace.

        Parameters
        ----------
        step           : current simulation step (integer index into price arrays).
        t              : elapsed seconds since session start. t = step * dt.
        resting_orders : book.mm_resting_orders — dict of currently posted MM orders.
                         Format: {order_id: {"price", "direction", "level", "age"}}

        Returns
        -------
        new_quotes : Quote objects to post to the book. Empty = do nothing.
        cancel_ids : order IDs to cancel before posting new_quotes.

        Decision logic (in priority order)
        -----------------------------------
        1. First call ever -> full 20-order ladder, no cancels.
        2. FX session boundary -> cancel all, full 20-order ladder.
        3. Best-price drift > threshold -> cancel ALL resting orders, full reprice.
        4. Full fill received -> replace only the filled (direction, level) slot.
        5. Inventory moved > inventory_requote_fraction since last reprice -> full reprice.
        6. Stale + stressed inventory -> cancel stale orders, reprice those slots.
        7. Nothing triggered -> return [], [] (no action).
        """

        # Step 0: update incremental EWMA vol estimate for this step
        self._update_ewma_vol(step)

        # Step 1: drain both fill queues (accumulated since last call via on_fill)
        filled_slots = {(f.direction, f.level) for f in self._pending_fills}
        self._pending_fills.clear()

        topup_ids = {e.order_id for e in self._pending_topups}
        self._pending_topups.clear()

        # Step 2: compute current theoretical best bid and ask
        bid_B, ask_B = self._get_stale_quotes(self.market_B, step, self._lag_B)
        bid_C, ask_C = self._get_stale_quotes(self.market_C, step, self._lag_C)
        # Cache stale quotes so execute_hedge can reuse them without a second lookup.
        self._cache_bid_B = bid_B
        self._cache_ask_B = ask_B
        self._cache_bid_C = bid_C
        self._cache_ask_C = ask_C
        fair_mid = (max(bid_B, bid_C) + min(ask_B, ask_C)) / 2.0

        sigma = self._estimate_vol(step)
        self._cache_sigma = sigma   # reused by execute_hedge

        inv_horizon_y = getattr(self, '_inv_horizon_y', 1.0 / (self.cfg.omega * TRADING_SECONDS_PER_YEAR))
        if self._hedge_emergency:
            inv_horizon_y *= self.cfg.emergency_penalty_multiplier

        # Session-adaptive k
        k_eff = self._session_k(t)

        # ── Spread first (needed to scale the reservation price shift) ──────────
        # A-S optimal spread under infinite horizon
        two_over_gamma = getattr(self, '_two_over_gamma', 2.0 / self.cfg.gamma)
        spread_AS_bps = (
            self.cfg.gamma * (sigma * 100.0) ** 2 * inv_horizon_y
            + two_over_gamma * np.log(1.0 + self.cfg.gamma / k_eff)
        )
        spread_AS = spread_AS_bps / 10_000.0 * fair_mid
        latency_coeff = getattr(self, '_latency_coeff',
                                2.0 * np.sqrt(self._effective_gap_s / TRADING_SECONDS_PER_YEAR))
        spread_latency = latency_coeff * sigma   # = 2·σ·√(gap/TSPY)
        inventory_ratio = self.inventory * getattr(self, '_inv_capital_K', 1.0 / self.capital_K)
        spread_inventory = self.cfg.alpha_spread * inventory_ratio**2 * spread_AS
        total_spread = spread_AS + spread_latency + spread_inventory
        half_spread = max(total_spread / 2.0, self.cfg.tick_size)

        # ── Reservation price ────────────────────────────────────────────────────
        # The original A-S formula (inventory × γ × σ² × inv_horizon_y) produces a
        # shift of ~530 bps at 10% inventory — orders of magnitude larger than the
        # half-spread (~3 bps) — because inventory is in raw EUR (0-900k).
        # Rescaled form: shift = inventory_ratio × half_spread, so that at 100%
        # inventory the quoted mid is displaced by ±1 half-spread from fair_mid.
        # Direction is correct (long → shift down → attract sells); magnitude is now
        # calibrated relative to the spread, with no new parameters.
        reservation_price = fair_mid - inventory_ratio * half_spread

        # Order-flow imbalance tilt (Cartea & Jaimungal)
        imbalance = self._compute_imbalance()
        reservation_price += self.cfg.alpha_imbalance * imbalance * fair_mid

        # ── Asymmetric delta skew (Guéant) ────────────────────────────────────────
        if self.cfg.use_asymmetric_delta:
            skew_delta = self.inventory * np.sqrt(self.cfg.gamma * sigma**2 / (2.0 * k_eff))
            skew_delta = np.clip(skew_delta, -(half_spread - 0.5 * self.cfg.tick_size),
                                              (half_spread - 0.5 * self.cfg.tick_size))
        else:
            skew_delta = 0.0

        best_bid = self._snap_to_tick(reservation_price - half_spread - skew_delta)
        best_ask = self._snap_to_tick(reservation_price + half_spread - skew_delta)
        if best_bid >= best_ask:
            best_ask = best_bid + self.cfg.tick_size

        threshold = self.cfg.requote_threshold_spread_fraction * total_spread

        # Priority 1: very first call. Full ladder, no prior state to compare against
        if self._first_quote:
            self._first_quote = False
            self._prev_best_bid = best_bid
            self._prev_best_ask = best_ask
            self._prev_inventory = self.inventory
            self._last_requote_rule = 1
            return self._build_ladder(best_bid, best_ask, inventory_ratio), []

        # Priority 2: FX session boundary — cancel everything, start fresh
        if self._is_session_reset(t):
            self._last_session_reset_t = t
            self._prev_best_bid = best_bid
            self._prev_best_ask = best_ask
            self._prev_inventory = self.inventory
            self._flow_history.clear()
            cancel_ids = list(resting_orders.keys())
            self._last_requote_rule = 2
            return self._build_ladder(best_bid, best_ask, inventory_ratio), cancel_ids

        # Priority 3: best price has drifted beyond threshold — cancel all and reprice
        price_drifted = (
            abs(best_bid - self._prev_best_bid) > threshold
            or abs(best_ask - self._prev_best_ask) > threshold
        )
        if price_drifted:
            cancel_ids = list(resting_orders.keys())
            self._prev_best_bid = best_bid
            self._prev_best_ask = best_ask
            self._prev_inventory = self.inventory
            self._last_requote_rule = 3
            return self._build_ladder(best_bid, best_ask, inventory_ratio), cancel_ids

        # Priority 4: full fills + partial top-ups. Handle both in one pass
        topup_orders = self._build_topups(resting_orders, topup_ids)
        if filled_slots or topup_orders:
            self._prev_best_bid = best_bid
            self._prev_best_ask = best_ask
            self._prev_inventory = self.inventory
            self._last_requote_rule = 4
            return self._build_partial_ladder(best_bid, best_ask, inventory_ratio, filled_slots) + topup_orders, []

        # Priority 5: inventory has drifted enough since the last reprice to invalidate the current skew, even if market prices have not moved (silent accumulation).
        inventory_moved = abs(self.inventory - self._prev_inventory) / self.capital_K # This ensures resting orders always reflect the current reservation price skew
        if inventory_moved > self.cfg.inventory_requote_fraction:
            cancel_ids = list(resting_orders.keys())
            self._prev_best_bid = best_bid
            self._prev_best_ask = best_ask
            self._prev_inventory = self.inventory
            self._last_requote_rule = 5
            return self._build_ladder(best_bid, best_ask, inventory_ratio), cancel_ids

        # Priority 6: stale orders — reprice aggressively regardless of inventory
        stale_slots: Set[Tuple[str, int]] = set()
        cancel_ids_6: List[str] = []
        for oid, info in resting_orders.items():
            if info["age"] > self._stale_steps:
                cancel_ids_6.append(oid)
                stale_slots.add((info["direction"], info["level"]))
        if stale_slots:
            self._prev_best_bid = best_bid
            self._prev_best_ask = best_ask
            self._prev_inventory = self.inventory
            self._last_requote_rule = 6
            return self._build_ladder_tight(best_bid, best_ask, inventory_ratio,
                                            self.cfg.stale_tight_fraction), cancel_ids_6

        # Nothing triggered. Resting orders are still valid, do nothing
        self._last_requote_rule = 0
        return [], []


    #  RISK MANAGEMENT

    def needs_hedge(self, fair_mid: float = 1.0) -> bool:
        """
        Hedge when the EUR inventory OR the USD notional exposure exceeds its limit.

        Capital is assumed to be allocated 50/50 between EUR and USD at inception,
        so each currency's limit is  delta_limit × (capital_K / 2).

            EUR check: |inventory|           > delta_limit × (K / 2)
            USD check: |inventory × price|   > delta_limit × (K / 2)
        """
        half_K = self.capital_K * 0.5
        limit = self.cfg.delta_limit * half_K
        return (abs(self.inventory) > limit or
                abs(self.inventory * fair_mid) > limit)

    def hedge_order(self, depth_B: float, depth_C: float, fair_mid: float,
                    sigma: float = 0.0) -> Tuple[float, float, float]:
        """
        Compute the optimal hedge split across B and C given actual available depth.

        Score formula (higher = preferred venue):
            score = 1 / (taker_fee + latency_cost + impact_factor / depth)

        where:
            latency_cost = vol_per_s × latency_s   (expected adverse move during execution lag)
            impact_factor = fair_mid × 0.0001       (Kyle-lambda proxy)

        The total hedge is capped at depth_B + depth_C: if that is less than |inventory|,
        we hedge as much as available and the caller handles the residual.
        Overflow from the preferred venue is rerouted to the other.
        """
        total_target = -self.inventory
        total_available = depth_B + depth_C
        total_size = float(np.sign(total_target)) * min(abs(total_target), total_available)

        if total_size == 0.0:
            return 0.0, 0.0, 0.0

        impact_factor = fair_mid * 0.0001
        vol_per_s = sigma / np.sqrt(TRADING_SECONDS_PER_YEAR) if sigma > 0.0 else 0.0

        score_B = (1.0 / (self.cfg.fee_B_taker + vol_per_s * self.cfg.latency_B_s
                          + impact_factor / max(depth_B, 1e-8))
                   if depth_B > 0.0 else 0.0)
        score_C = (1.0 / (self.cfg.fee_C_taker + vol_per_s * self.cfg.latency_C_s
                          + impact_factor / max(depth_C, 1e-8))
                   if depth_C > 0.0 else 0.0)

        total_score = score_B + score_C
        ratio_B = self.cfg.weight_B if total_score == 0.0 else score_B / total_score

        ideal_B = total_size * ratio_B
        size_B = float(np.sign(ideal_B)) * min(abs(ideal_B), depth_B)
        overflow = ideal_B - size_B
        ideal_C = total_size * (1.0 - ratio_B) + overflow
        size_C = float(np.sign(ideal_C)) * min(abs(ideal_C), depth_C) if ideal_C != 0.0 else 0.0

        size_B = round(size_B, 2)
        size_C = round(size_C, 2)

        fee_cost = (abs(size_B) * self.cfg.fee_B_taker * fair_mid
                    + abs(size_C) * self.cfg.fee_C_taker * fair_mid)
        return size_B, size_C, fee_cost

    def fill_cost(self, size: float, fair_mid: float) -> float:
        return abs(size) * self.cfg.fee_A_maker * fair_mid

    def execute_hedge(self, step: int, t: float) -> bool:
        """
        Execute the optimal hedge across B and C when inventory breaches delta_limit,
        or at EOD (every eod_flat_interval seconds) regardless of inventory size.

        Depth is read from market_B.depth / market_C.depth at the lagged step (same
        latency as prices). If depth arrays are not generated, falls back to capital_K
        scaled by volume weight so the hedge always executes.

        Fill price: taker order on B or C crosses the spread — buying fills at the venue
        ask, selling fills at the venue bid (both at the lagged step). Taker fee is then
        applied on top of that fill price.

        Three outcomes based on post-hedge inventory ratio:
          ≤ hedge_partial_limit   : full or partial hedge succeeded — clear emergency flag.
          > hedge_partial_limit   : available depth was insufficient — set _hedge_emergency
                                    so compute_quotes applies an extreme penalty to force
                                    asymmetric quotes on A toward inventory reduction.
        """
        # Use B/C cached fair mid (set by compute_quotes this step) — more accurate
        # than mid_A for valuing the USD notional of our EUR inventory.
        fair_mid = (
            max(self._cache_bid_B, self._cache_bid_C) +
            min(self._cache_ask_B, self._cache_ask_C)
        ) / 2.0

        # EOD flat: force inventory to zero each time we cross a day boundary.
        force_flat = False
        if self.cfg.eod_flat_interval > 0.0 and abs(self.inventory) > 0.0:
            current_day = int(t // self.cfg.eod_flat_interval)
            if current_day > self._last_eod_day:
                self._last_eod_day = current_day
                force_flat = True

        if not force_flat and not self.needs_hedge(fair_mid):
            self._hedge_emergency = False
            return False

        stale_B = max(0, step - self._lag_B)
        stale_C = max(0, step - self._lag_C)
        depth_B = (float(self.market_B.depth[stale_B])
                   if self.market_B.depth is not None
                   else self.capital_K * self.cfg.weight_B)
        depth_C = (float(self.market_C.depth[stale_C])
                   if self.market_C.depth is not None
                   else self.capital_K * self.cfg.weight_C)

        # Reuse stale quotes and sigma already computed by compute_quotes this step.
        bid_B = getattr(self, '_cache_bid_B', None)
        if bid_B is None:
            bid_B, ask_B = self._get_stale_quotes(self.market_B, step, self._lag_B)
            bid_C, ask_C = self._get_stale_quotes(self.market_C, step, self._lag_C)
            sigma = self._estimate_vol(step)
        else:
            ask_B = self._cache_ask_B
            bid_C = self._cache_bid_C
            ask_C = self._cache_ask_C
            sigma = getattr(self, '_cache_sigma', self._estimate_vol(step))

        size_B, size_C, _ = self.hedge_order(depth_B, depth_C, fair_mid, sigma)

        inventory_after = self.inventory + size_B + size_C
        partial_limit = self.cfg.hedge_partial_limit * self.capital_K * 0.5
        if abs(inventory_after) > partial_limit:
            self._hedge_emergency = True
        else:
            self._hedge_emergency = False

        for size, venue, fee_rate, bid_v, ask_v in [
            (size_B, "B", self.cfg.fee_B_taker, bid_B, ask_B),
            (size_C, "C", self.cfg.fee_C_taker, bid_C, ask_C),
        ]:
            if size == 0.0:
                continue
            abs_size = abs(size)
            direction = "buy" if size > 0 else "sell"
            # Taker crosses the spread: buying hits the ask, selling hits the bid
            fill_price = ask_v if size > 0 else bid_v
            fee_cost = abs_size * fill_price * fee_rate
            cash_flow = (-abs_size * fill_price * (1.0 + fee_rate)
                         if size > 0
                         else abs_size * fill_price * (1.0 - fee_rate))
            self.inventory += size
            self._fill_history.append({
                "order_id": None,
                "level": 0,
                "step": step,
                "t": t,
                "direction": direction,
                "price": fill_price,
                "size": abs_size,
                "is_full_fill": True,
                "fair_mid": fair_mid,
                "fee_cost": fee_cost,
                "cash_flow": cash_flow,
                "inventory_after": self.inventory,
                "is_hedge": True,
                "is_eod_flat": force_flat,
                "venue": venue,
            })

        return size_B != 0.0 or size_C != 0.0


    #  DIAGNOSTICS

    def snapshot(self, step: int, t: float) -> dict:
        bid_B, ask_B = self._get_stale_quotes(self.market_B, step, self._lag_B)
        bid_C, ask_C = self._get_stale_quotes(self.market_C, step, self._lag_C)
        best_bid_ref = max(bid_B, bid_C)
        best_ask_ref = min(ask_B, ask_C)
        fair_mid = (best_bid_ref + best_ask_ref) / 2.0

        sigma = self._estimate_vol(step)
        inv_horizon_y = 1.0 / (self.cfg.omega * TRADING_SECONDS_PER_YEAR)
        k_eff = self._session_k(t)

        spread_AS_bps = (
            self.cfg.gamma * (sigma * 100.0) ** 2 * inv_horizon_y
            + (2.0 / self.cfg.gamma) * np.log(1.0 + self.cfg.gamma / k_eff)
        )
        spread_AS = spread_AS_bps / 10_000.0 * fair_mid
        sigma_per_s = sigma / np.sqrt(TRADING_SECONDS_PER_YEAR)
        spread_latency = 2.0 * sigma_per_s * np.sqrt(self._effective_gap_s)
        inventory_ratio = self.inventory / self.capital_K
        spread_inventory = self.cfg.alpha_spread * inventory_ratio**2 * spread_AS
        total_spread = spread_AS + spread_latency + spread_inventory
        half_spread = max(total_spread / 2.0, self.cfg.tick_size)

        reservation_price = fair_mid - inventory_ratio * half_spread
        imbalance = self._compute_imbalance()
        reservation_price += self.cfg.alpha_imbalance * imbalance * fair_mid

        if self.cfg.use_asymmetric_delta:
            skew_delta = self.inventory * np.sqrt(
                self.cfg.gamma * sigma**2 / (2.0 * k_eff)
            )
            max_skew = half_spread - 0.5 * self.cfg.tick_size
            skew_delta = np.clip(skew_delta, -max_skew, max_skew)
        else:
            skew_delta = 0.0

        ask_half = half_spread - skew_delta
        bid_half = half_spread + skew_delta
        best_bid = self._snap_to_tick(reservation_price - bid_half)
        best_ask = self._snap_to_tick(reservation_price + ask_half)

        return {
            "step": step,
            "t_seconds": t,
            "bid_B": bid_B, "ask_B": ask_B,
            "bid_C": bid_C, "ask_C": ask_C,
            "fair_mid": fair_mid,
            "reservation_price": reservation_price,
            "inv_horizon_y": inv_horizon_y,
            "k_eff": k_eff,
            "spread_AS": spread_AS,
            "spread_latency": spread_latency,
            "spread_inventory": spread_inventory,
            "total_spread": total_spread,
            "half_spread": half_spread,
            "skew_delta": skew_delta,
            "ask_half": ask_half,
            "bid_half": bid_half,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": (best_ask - best_bid) / fair_mid * 10_000.0,
            "inventory_EUR": self.inventory,
            "inventory_ratio": inventory_ratio,
            "needs_hedge": self.needs_hedge(fair_mid),
            "effective_gap_ms": self._effective_gap_s * 1000.0,
            "sigma_used": sigma,
            "n_fills": len(self._fill_history),
            "imbalance": imbalance,
        }

    @property
    def trade_history(self):
        """All fills (partial + full) as a DataFrame. Use for P&L analysis."""
        if not self._fill_history:
            return pd.DataFrame(columns=["order_id", "level", "step", "t", "direction",
                                         "price", "size", "is_full_fill", "fair_mid",
                                         "fee_cost", "cash_flow", "inventory_after",
                                         "is_hedge", "is_eod_flat", "venue"])
        return pd.DataFrame(self._fill_history)


    #  PRIVATE HELPERS

    def _build_ladder(self, best_bid: float, best_ask: float, inventory_ratio: float) -> List[Order]:
        """Build the full 10-level bid/ask ladder. Used on first quote and full reprices."""
        inventory_skew = np.clip(inventory_ratio, -1.0, 1.0)
        orders: List[Order] = []
        for i in range(1, self.cfg.n_levels + 1):
            base_size = self.cfg.Q_base * np.exp(-self.cfg.beta * i)
            bid_size = round(base_size * (1.0 - 0.5 * inventory_skew))
            ask_size = round(base_size * (1.0 + 0.5 * inventory_skew))
            bid_price = self._snap_to_tick(best_bid - (i - 1) * self.cfg.tick_size)
            ask_price = self._snap_to_tick(best_ask + (i - 1) * self.cfg.tick_size)
            orders.append(Order(generate_order_id(), "buy", bid_price, bid_size, "limit_order", "market_maker", i))
            orders.append(Order(generate_order_id(), "sell", ask_price, ask_size, "limit_order", "market_maker", i))
        return orders

    def _build_ladder_tight(self, best_bid: float, best_ask: float, inventory_ratio: float, spread_multiplier: float) -> List[Order]:
        """Full ladder repriced with a tighter spread (spread_multiplier < 1.0).
        Squeezes the bid/ask around the current mid by spread_multiplier before building."""
        mid = (best_bid + best_ask) * 0.5
        half = (best_ask - best_bid) * 0.5 * spread_multiplier
        tight_bid = self._snap_to_tick(mid - half)
        tight_ask = self._snap_to_tick(mid + half)
        if tight_bid >= tight_ask:
            tight_ask = tight_bid + self.cfg.tick_size
        return self._build_ladder(tight_bid, tight_ask, inventory_ratio)

    def _build_partial_ladder(self, best_bid: float, best_ask: float, inventory_ratio: float, slots: Set[Tuple[str, int]]) -> List[Order]:
        """
        Surgical rebuild: only emit quotes for the exact (direction, level) slots cancelled.
        e.g. slots={("sell", 1)} → returns a single new ask at level 1 only.
        """
        inventory_skew = np.clip(inventory_ratio, -1.0, 1.0)
        orders: List[Order] = []
        for i in range(1, self.cfg.n_levels + 1):
            base_size = self.cfg.Q_base * np.exp(-self.cfg.beta * i)
            if ("buy", i) in slots:
                bid_size = round(base_size * (1.0 - 0.5 * inventory_skew))
                bid_price = self._snap_to_tick(best_bid - (i - 1) * self.cfg.tick_size)
                orders.append(Order(generate_order_id(), "buy", bid_price, bid_size, "limit_order", "market_maker", i))
            if ("sell", i) in slots:
                ask_size = round(base_size * (1.0 + 0.5 * inventory_skew))
                ask_price = self._snap_to_tick(best_ask + (i - 1) * self.cfg.tick_size)
                orders.append(Order(generate_order_id(), "sell", ask_price, ask_size, "limit_order", "market_maker", i))
        return orders

    def _build_topups(self, resting_orders: dict, order_ids: set) -> List[Order]:
        """
        For each partially filled order still resting, emit a complementary order
        at the same price for the missing size (original_size - remaining_size).
        No cancel is needed — this just tops the level back up to its original depth.
        """
        orders: List[Order] = []
        for oid in order_ids:
            if oid not in resting_orders:
                continue
            info = resting_orders[oid]
            topup_size = round(info["original_size"] - info["remaining_size"])
            if topup_size <= 0:
                continue
            orders.append(Order(
                generate_order_id(),
                info["direction"],
                info["price"],
                topup_size,
                "limit_order",
                "market_maker",
                info["level"],
            ))
        return orders

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

    def _session_k(self, t: float) -> float:
        """
        Return the session-adjusted order-arrival intensity k for time t.
        Tokyo (00:00-08:00 UTC) is the thinnest session; London (08:00-16:00) is the
        deepest. Using a single k across all sessions misprice fill probability:
        the same quoted distance attracts very different order flow in each regime.
        """
        t_mod = t % (24 * 3600)   # normalise to position within a 24h day
        if t_mod < 8 * 3600:
            return self.cfg.k * _SESSION_K_MULTIPLIERS["tokyo"]
        elif t_mod < 16 * 3600:
            return self.cfg.k * _SESSION_K_MULTIPLIERS["london"]
        else:
            return self.cfg.k * _SESSION_K_MULTIPLIERS["newyork"]

    def _update_ewma_vol(self, step: int) -> None:
        """
        Incremental EWMA variance update — O(1) per step.
        Call exactly once per step, before _estimate_vol().
        Uses a single log-return from the previous step to current step.
        _ewma_prev_price caches the last p1 so we avoid reading index step-1 twice.
        """
        p1 = float(self.market_B.noised_mid_price[step])
        if self._ewma_prev_price is None:
            self._ewma_prev_price = p1
            return
        p0 = self._ewma_prev_price
        self._ewma_prev_price = p1
        if p0 <= 0.0:
            return
        r = np.log(p1 / p0)
        alpha = 2.0 / (self._vol_window_steps + 1)
        if self._ewma_var is None:
            self._ewma_var = r * r
        else:
            self._ewma_var = alpha * r * r + (1.0 - alpha) * self._ewma_var

    def _estimate_vol(self, step: int) -> float:
        """
        Return the current annualised volatility estimate.
        Uses the incremental EWMA maintained by _update_ewma_vol().
        Falls back to the parametric vol below vol_min_steps.
        """
        if step < self._vol_min_steps or self._ewma_var is None:
            return self.market_B.stock.vol
        dt_frac = getattr(self, '_dt_frac', self.market_B.stock.time_step / TRADING_SECONDS_PER_YEAR)
        return max(np.sqrt(self._ewma_var / dt_frac), 1e-6)

    def _get_stale_quotes(self, market: Market, step: int, lag: int) -> Tuple[float, float]:
        stale_step = max(0, step - lag)
        return float(market.bid_price[stale_step]), float(market.ask_price[stale_step])

    def _compute_imbalance(self) -> float:
        """
        Order flow imbalance in [-1, 1] over the last imbalance_window fills.
        +1 = all client buys (hitting our ask) -> bullish signal.
        -1 = all client sells (hitting our bid) -> bearish signal.
        O(1): maintained incrementally via _imbalance_n_sells / _imbalance_n_buys.
        """
        total = self._imbalance_n_sells + self._imbalance_n_buys
        if total < self.cfg.imbalance_min_samples:
            return 0.0
        return (self._imbalance_n_sells - self._imbalance_n_buys) / total

    def _snap_to_tick(self, price: float) -> float:
        tick = self.cfg.tick_size
        return round(round(price / tick) * tick, 6)
