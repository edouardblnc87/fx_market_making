"""
Power-law order size distribution.

Theory (Avellaneda & Stoikov 2006, Section 2.5, eq. 2.8)
─────────────────────────────────────────────────────────
The density of market order sizes follows a power law:

    f^Q(x) ∝ x^{-1-α}

with α ≈ 1.5 (Gabaix et al., Gopikrishnan et al., Maslow & Mills).

We sample from a **truncated Pareto** distribution on [x_min, x_max] using
inverse-CDF transform so that sizes are bounded and well-behaved in simulation.
"""

import numpy as np


def sample_size(
    alpha: float,
    x_min: float,
    x_max: float,
    rng: np.random.Generator,
) -> int:
    """
    Draw one order size from a truncated Pareto distribution.

    CDF:  F(x) = 1 - (x_min / x)^α   for x ≥ x_min
    Truncated to [x_min, x_max]:

        F_trunc(x) = [1 - (x_min/x)^α] / [1 - (x_min/x_max)^α]

    Inverse CDF (used for sampling):

        x = x_min · (1  -  U · (1 - (x_min/x_max)^α))^{-1/α}

    where U ~ Uniform(0, 1).

    Parameters
    ----------
    alpha : tail exponent (> 0)
    x_min : minimum order size (truncation floor)
    x_max : maximum order size (truncation ceiling)
    rng   : numpy random Generator

    Returns
    -------
    size : sampled order size, rounded to nearest integer
    """
    u = rng.uniform()
    ratio = (x_min / x_max) ** alpha          # probability mass beyond x_max
    x = x_min * (1.0 - u * (1.0 - ratio)) ** (-1.0 / alpha)
    return max(int(round(x)), int(x_min))
