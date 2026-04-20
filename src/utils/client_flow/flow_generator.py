"""
Client order flow generator.

Produces a time-indexed stream of client orders using:
  - Poisson arrival with exponential intensity  λ(δ) = A·exp(-k·δ)
  - Power-law order sizes  f^Q(x) ∝ x^{-1-α}
  - 50/50 mix of market orders and limit orders (configurable)

Orders are returned as `Order` objects compatible with `Order_book.add_orders_batch`.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from ..order_book.order_book_impl import Order, _generate_order_id
from .arrival import intensity, sample_arrival_count
from .size_model import sample_size
from . import config as default_cfg


@dataclass
class ClientFlowConfig:
    """All tunable parameters for the client order flow model."""

    # Arrival intensity (per second): λ(δ) = A · exp(-k · δ)
    A_buy:    float = default_cfg.A_BUY
    A_sell:   float = default_cfg.A_SELL
    k_buy:    float = default_cfg.K_BUY
    k_sell:   float = default_cfg.K_SELL

    # Power-law size distribution
    alpha:    float = default_cfg.ALPHA
    size_min: float = default_cfg.SIZE_MIN
    size_max: float = default_cfg.SIZE_MAX

    # Order type mix
    market_order_ratio: float = default_cfg.MARKET_ORDER_RATIO


class ClientFlowGenerator:
    """
    Generate a realistic client order stream for Exchange A.

    At each simulation time step the generator:
      1. Computes the distance δ from mid to the MM's best bid/ask
      2. Evaluates the Poisson intensity λ(δ) for each side
      3. Samples whether a buy and/or sell arrival occurs
      4. For each arrival: draws a power-law size, flips market/limit,
         prices the order, and returns an Order object with origin="client"
    """

    def __init__(
        self,
        config: ClientFlowConfig | None = None,
        seed: int | None = None,
    ):
        """Initialise the generator with an optional config and random seed."""
        self.cfg = config if config is not None else ClientFlowConfig()
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Per-step generation
    # ------------------------------------------------------------------

    def generate_step(
        self,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        dt: float,
    ) -> List[Order]:
        """
        Generate client orders for a single simulation time step.

        Parameters
        ----------
        mid_price : current mid-price (e.g. from Market or Quoter)
        best_bid  : MM's best bid on Exchange A
        best_ask  : MM's best ask on Exchange A
        dt        : time step in seconds

        Returns
        -------
        orders : list of Order objects (N_buy + N_sell, Poisson-sampled per side)
        """
        orders: List[Order] = []

        # Distances from mid in basis points (paper: δ^b = s - p^b, δ^a = p^a - s)
        # Convert price units → bps so that k has meaningful scale for FX
        delta_bid_bps = max(mid_price - best_bid, 0.0) / mid_price * 10_000
        delta_ask_bps = max(best_ask - mid_price, 0.0) / mid_price * 10_000

        # Buy side: N_buy ~ Poisson(λ^a(δ_ask) · dt)
        lambda_buy = intensity(delta_ask_bps, self.cfg.A_buy, self.cfg.k_buy)
        n_buy = sample_arrival_count(lambda_buy, dt, self._rng)
        for _ in range(n_buy):
            orders.append(self._build_order("buy", mid_price, best_bid, best_ask))

        # Sell side: N_sell ~ Poisson(λ^b(δ_bid) · dt)
        lambda_sell = intensity(delta_bid_bps, self.cfg.A_sell, self.cfg.k_sell)
        n_sell = sample_arrival_count(lambda_sell, dt, self._rng)
        for _ in range(n_sell):
            orders.append(self._build_order("sell", mid_price, best_bid, best_ask))

        return orders

    # ------------------------------------------------------------------
    # Full-session generation
    # ------------------------------------------------------------------

    def generate_session(
        self,
        mid_prices: np.ndarray,
        bid_prices: np.ndarray,
        ask_prices: np.ndarray,
        dt: float,
    ) -> List[Tuple[int, Order]]:
        """
        Generate all client orders for a full trading session.

        Parameters
        ----------
        mid_prices : 1D array (N+1,) — mid-price at each step
        bid_prices : 1D array (N+1,) — MM best bid at each step
        ask_prices : 1D array (N+1,) — MM best ask at each step
        dt         : time step in seconds

        Returns
        -------
        session_orders : list of (step_index, Order) tuples, sorted by step
        """
        session_orders: List[Tuple[int, Order]] = []

        for step in range(len(mid_prices)):
            step_orders = self.generate_step(
                mid_price=float(mid_prices[step]),
                best_bid=float(bid_prices[step]),
                best_ask=float(ask_prices[step]),
                dt=dt,
            )
            for order in step_orders:
                session_orders.append((step, order))

        return session_orders

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_order(
        self,
        direction: str,
        mid_price: float,
        best_bid: float,
        best_ask: float,
    ) -> Order:
        """
        Build a single client Order with random type (market/limit) and
        power-law size.
        """
        size = sample_size(
            self.cfg.alpha, self.cfg.size_min, self.cfg.size_max, self._rng,
        )

        is_market = self._rng.uniform() < self.cfg.market_order_ratio

        if is_market:
            order_type = "market_order"
            # Market orders cross immediately: buy at ask, sell at bid
            if direction == "buy":
                price = best_ask
            else:
                price = best_bid
        else:
            order_type = "limit_order"
            # Limit orders sit in the book at a distance δ from mid
            # δ drawn from Exponential(k) in bps, then converted to price units
            if direction == "buy":
                k = self.cfg.k_buy
                delta_bps = self._rng.exponential(1.0 / k) if k > 0 else 0.0
                price = mid_price - delta_bps / 10_000 * mid_price
            else:
                k = self.cfg.k_sell
                delta_bps = self._rng.exponential(1.0 / k) if k > 0 else 0.0
                price = mid_price + delta_bps / 10_000 * mid_price

        # Round to 4 decimal places (tick convention)
        price = round(price, 4)

        return Order(
            id=_generate_order_id(),
            direction=direction,
            price=price,
            size=size,
            type=order_type,
            origin="client",
        )
