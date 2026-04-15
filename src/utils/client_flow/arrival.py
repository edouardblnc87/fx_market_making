"""
Poisson arrival model for client orders.

Theory (Avellaneda & Stoikov 2006, Section 2.5)
─────────────────────────────────────────────────
Market buy/sell orders arrive at the dealer's quotes with Poisson intensity

    λ(δ) = A · exp(-k · δ)

where δ ≥ 0 is the distance between the dealer's quote and the mid-price,
expressed in **basis points** (1 bp = 0.01%).
Larger δ → fewer arrivals (quotes far from mid are rarely lifted/hit).

A and k are derived from:
  - Λ : overall frequency of market orders (per second)
  - α : power-law tail exponent of order sizes  (f^Q ∝ x^{-1-α})
  - K : log-impact constant  (Δp ∝ ln Q  →  K converts to exp decay)

  A = Λ / α ,   k = α · K          (paper eq. 2.11)
"""

import numpy as np


def intensity(delta_bps: float, A: float, k: float) -> float:
    """
    Compute Poisson arrival intensity λ(δ) = A · exp(-k · δ).

    Parameters
    ----------
    delta_bps : distance from mid-price in basis points (≥ 0)
                1 bp = 0.0001 for a price near 1.0 (EUR/USD)
    A         : baseline arrival rate (per second) at δ = 0
    k         : exponential decay rate (per bp)

    Returns
    -------
    λ : arrival intensity (per second)
    """
    return A * np.exp(-k * delta_bps)


def sample_arrival(lambda_val: float, dt: float, rng: np.random.Generator) -> bool:
    """
    Sample whether a Poisson event occurs in a time interval dt.

    Uses the exact probability  P(N ≥ 1) = 1 - exp(-λ·dt)  rather than
    the Bernoulli approximation  P ≈ λ·dt  (which breaks when λ·dt is not small).

    Parameters
    ----------
    lambda_val : Poisson intensity (per second)
    dt         : time step (seconds)
    rng        : numpy random Generator for reproducibility

    Returns
    -------
    True if at least one arrival occurs in [t, t+dt)
    """
    prob = 1.0 - np.exp(-lambda_val * dt)
    return rng.uniform() < prob


def sample_arrival_count(lambda_val: float, dt: float, rng: np.random.Generator) -> int:
    """
    Sample the number of Poisson arrivals in a time interval dt.

    N ~ Poisson(λ · dt)

    Parameters
    ----------
    lambda_val : Poisson intensity (per second)
    dt         : time step (seconds)
    rng        : numpy random Generator for reproducibility

    Returns
    -------
    Number of arrivals in [t, t+dt)
    """
    return int(rng.poisson(lambda_val * dt))
