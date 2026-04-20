"""Spread utility functions: rolling realized volatility estimator and asymmetric spread evolution."""

import numpy as np
from ..stock_simulation.config import TRADING_SECONDS_PER_DAY

# Number of trading days per year — standard equity convention
TRADING_DAYS_PER_YEAR = 252


def compute_rv_zero_mean(true_path: np.ndarray, window_size: int, dt: float) -> np.ndarray:
    """
    Compute annualized rolling realized volatility using the zero-mean estimator.

    Why zero-mean instead of sample std?
    ─────────────────────────────────────
    The standard compute_realized_volatility in Stock uses np.std, which subtracts
    the sample mean of log-returns before squaring. Over short windows (e.g. 10–5000
    steps), the sample mean is just noise — subtracting it adds estimation error
    rather than removing bias. The zero-mean estimator RVar(t) = mean(r²) is both
    unbiased and lower-variance when the true drift is zero (or near-zero), which
    matches our GBM with mu=0.

    Why use the true GBM path and not the noised mid?
    ──────────────────────────────────────────────────
    The noised mid price = V + microstructure noise. Computing vol from it would
    inflate the estimate because the noise variance adds on top of the true variance.
    We have V directly in simulation so we use it.

    Parameters
    ----------
    true_path   : 1D array shape (N+1,) — the true GBM price path (stock.simulation)
    window_size : number of past steps to include in each vol estimate
    dt          : time step in seconds

    Returns
    -------
    rv_ann : 1D array shape (N+1,) — annualized realized vol at each step
    """

    # Log-returns from the true price path, shape (N,)
    log_returns = np.diff(np.log(true_path))

    # Theoretical per-step variance used to fill the warmup period (no look-ahead)
    # sigma_step² = (annualized vol)² * dt/T_year, but here we derive it from the
    # returns directly: average r² over the full path is our best prior at t=0
    sigma_step_sq = np.mean(log_returns ** 2)

    # Annualization constant: scales per-step vol up to annual vol
    # C_ann = sqrt(seconds_per_day * trading_days / dt)
    C_ann = np.sqrt(TRADING_SECONDS_PER_DAY * TRADING_DAYS_PER_YEAR / dt)

    rv_ann = np.zeros(len(true_path))  # shape (N+1,), index 0 is t=0

    # Warmup period [0, window_size): no history yet, fill with the theoretical
    # per-step vol scaled to annual — this is the prior a market maker would use
    rv_ann[:window_size] = np.sqrt(sigma_step_sq) * C_ann

    # Vectorized rolling sum using cumulative sum — O(N) regardless of window_size.
    # cumr2[i+1] - cumr2[i-w+1] = sum of r^2 over [i-w+1, i] (w elements)
    r2    = log_returns ** 2
    cumr2 = np.empty(len(r2) + 1)
    cumr2[0] = 0.0
    np.cumsum(r2, out=cumr2[1:])
    rvar = (cumr2[window_size:] - cumr2[:-window_size]) / window_size
    rv_ann[window_size:] = np.sqrt(rvar) * C_ann

    return rv_ann


def evolve_s_excess(
    s_star:  np.ndarray,
    kappa_u: float,
    kappa_d: float,
    dt:      float,
    sigma_s: float = 0.0,
) -> np.ndarray:
    """
    Evolve the excess spread S_excess forward in time using asymmetric mean reversion.

    This step is inherently sequential — each value depends on the previous one —
    so it cannot be vectorized. The asymmetry is: kappa_u >> kappa_d, meaning
    the spread widens fast (fear of adverse selection) and tightens slowly
    (competitive pressure only arrives after confirmed calm).

    Discrete-time update rule:
        S_excess[t+1] = (1 - kappa*dt) * S_excess[t] + kappa*dt * S_star[t]
        S_excess[t+1] = max(S_excess[t+1], 0)   ← floor: spread never below S_0

    Stability condition: kappa * dt < 1 must hold, otherwise the process overshoots.

    Parameters
    ----------
    s_star  : 1D array (N+1,) — the vol-driven target excess spread at each step
    kappa_u : upward reversion speed (per second) — used when target > current
    kappa_d : downward reversion speed (per second) — used when target <= current
    dt      : time step in seconds
    sigma_s : optional spread noise vol (set 0 to disable) — adds small random
              fluctuations to make the spread path look less perfectly smooth

    Returns
    -------
    s_excess : 1D array (N+1,) — the realized excess spread path
    """

    N        = len(s_star)
    s_excess = np.zeros(N)

    # Pre-draw noise if requested — independent of price process
    if sigma_s > 0:
        noise = sigma_s * np.sqrt(dt) * np.random.standard_normal(N)
    else:
        noise = np.zeros(N)

    # Sequential forward pass — cannot be parallelized
    for t in range(N - 1):

        # Regime switch: widening (fear) vs tightening (competition)
        if s_star[t] > s_excess[t]:
            kappa = kappa_u     # target above current → widen fast
        else:
            kappa = kappa_d     # target below current → tighten slowly

        # Weighted average between current excess and target
        s_excess[t + 1] = (1 - kappa * dt) * s_excess[t] + kappa * dt * s_star[t]

        # Optional spread noise — makes path look realistically ragged
        s_excess[t + 1] += noise[t]

        # Floor enforcement: excess spread can never go negative
        # (competition prevents spread from falling below minimum cost)
        s_excess[t + 1] = max(s_excess[t + 1], 0.0)

    return s_excess
