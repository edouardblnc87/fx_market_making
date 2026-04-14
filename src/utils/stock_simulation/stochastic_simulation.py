import numpy as np
from .config import *



def generate_time_grid(n_days: int = 1, dt_seconds: float = 0.05) -> np.ndarray:
    """
    Return a discrete time grid for an intraday stochastic simulation.

    Parameters
    ----------
    n_days : int
        Number of trading sessions to simulate (default 1).
    dt : float
        Time step in seconds (default 0.05 s = 50 ms).

    """


    T = n_days * TRADING_SECONDS_PER_DAY
    N = round(T / dt_seconds)
    return np.linspace(0.0, T, N + 1)

def generate_gbm_path(
    time_grid:     np.ndarray,
    S0:            float = 100.0,
    drift:         float = 0.0,   # annualized drift μ  (e.g. 0.05 = 5 %/year)
    vol_annualized: float = 0.20,  # annualized vol  σ  (e.g. 0.20 = 20 %/year)
    tick_size:     float = 0.0001,
) -> np.ndarray:
    """
    Simulate a Geometric Brownian Motion price path on a seconds-based time grid.

    Model:  dS = μ S dt + σ S dW

    Parameters are passed in annualized terms (market convention: vol is always
    quoted annualized). They are scaled internally to the time step:

        σ_dt = σ_annual * sqrt(dt / T_year_s)          # one-step std of log-return
        μ_dt = (μ_annual - ½σ_annual²) * dt / T_year_s # Itô correction included

    Parameters
    ----------
    time_grid      : output of generate_time_grid(), in seconds.
    S0             : initial stock price.
    drift          : annualized drift μ (risk-neutral → 0, historical → ~0.05–0.10).
    vol_annualized : annualized volatility σ (e.g. 0.20 for 20 %).
    tick_size      : price grid resolution; set to 0 to disable rounding.

    Returns
    -------
    np.ndarray, shape (N+1,)
        Simulated price path, aligned with time_grid.
    """
    dt = time_grid[1] - time_grid[0]   # constant step in seconds
    N  = len(time_grid) - 1

    # ── Scale annualized parameters to one time step ──────────────────────────
    # Variance is proportional to time → σ scales with sqrt(time ratio)
    sigma_dt = vol_annualized * np.sqrt(dt / TRADING_SECONDS_PER_YEAR)
    # Drift in log-space already includes the Itô ½σ² correction
    mu_dt    = (drift - 0.5 * vol_annualized ** 2) * (dt / TRADING_SECONDS_PER_YEAR)

    # ── Draw increments and build cumulative log-price path ───────────────────
    Z        = np.random.standard_normal(N)
    vol_realized = sigma_dt * Z  

    log_rets = mu_dt + vol_realized                    # shape (N,)
    log_S = np.log(S0) + np.concatenate([[0.0], np.cumsum(log_rets)])

    S = np.exp(log_S)

    # ── Snap to tick grid ─────────────────────────────────────────────────────
    if tick_size > 0:
        S = np.round(S / tick_size) * tick_size

    return S, vol_realized, mu_dt, dt, N
