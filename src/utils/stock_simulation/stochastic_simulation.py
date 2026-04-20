"""Stochastic price path generators: GBM, Heston, and GARCH(1,1) with optional jumps."""

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


def generate_heston_path(
    time_grid:  np.ndarray,
    S0:         float = 100.0,
    drift:      float = 0.0,
    v0:         float = 0.04,    # initial variance (= vol² = 0.20² for 20%)
    kappa:      float = 2.0,     # mean-reversion speed
    theta:      float = 0.04,    # long-run variance (= target vol²)
    xi:         float = 0.3,     # vol of vol
    rho:        float = -0.1,    # spot-vol correlation (FX: mild negative)
    tick_size:  float = 0.0001,
) -> tuple:
    """
    Simulate a Heston (1993) stochastic volatility price path.

    Model:
        dS = μ S dt + √v S dW_S
        dv = κ(θ - v) dt + ξ √v dW_v
        corr(dW_S, dW_v) = ρ dt

    Discretization: Full-truncation Euler-Maruyama (Lord et al. 2010).

    Returns
    -------
    S             : np.ndarray, shape (N+1,) — price path
    v             : np.ndarray, shape (N+1,) — instantaneous variance path
    vol_realized  : np.ndarray, shape (N,)   — per-step log-return residuals
    dt            : float — time step in seconds
    N             : int   — number of steps
    """
    dt = time_grid[1] - time_grid[0]
    N  = len(time_grid) - 1

    # ── Feller condition check ────────────────────────────────────────────────
    feller = 2.0 * kappa * theta / (xi ** 2)
    if feller <= 1.0:
        import warnings
        warnings.warn(
            f"Feller condition violated (2κθ/ξ² = {feller:.3f} ≤ 1). "
            "The variance process may hit zero; results may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    dt_year = dt / TRADING_SECONDS_PER_YEAR
    rho_perp = np.sqrt(max(1.0 - rho ** 2, 0.0))  # sqrt(1 - ρ²) for Cholesky

    # ── Correlated Brownian increments (Cholesky) ─────────────────────────────
    Z_S   = np.random.standard_normal(N)
    Z_ind = np.random.standard_normal(N)
    Z_v   = rho * Z_S + rho_perp * Z_ind

    # ── Simulate variance (CIR) and price paths ───────────────────────────────
    v = np.empty(N + 1)
    S = np.empty(N + 1)
    v[0] = v0
    S[0] = S0

    vol_realized = np.empty(N)

    for t in range(N):
        v_plus = max(v[t], 0.0)                         # full-truncation
        sqrt_v_dt = np.sqrt(v_plus * dt_year)

        # variance update
        v[t + 1] = max(
            v[t] + kappa * (theta - v_plus) * dt_year + xi * sqrt_v_dt * Z_v[t],
            0.0,
        )

        # log-return increment
        log_ret = (drift - 0.5 * v_plus) * dt_year + sqrt_v_dt * Z_S[t]
        vol_realized[t] = sqrt_v_dt * Z_S[t]           # stochastic part only
        S[t + 1] = S[t] * np.exp(log_ret)

    # ── Snap to tick grid ─────────────────────────────────────────────────────
    if tick_size > 0:
        S = np.round(S / tick_size) * tick_size

    return S, v, vol_realized, dt, N


def generate_garch_path(
    time_grid:      np.ndarray,
    S0:             float = 100.0,
    drift:          float = 0.0,
    vol_annualized: float = 0.20,
    alpha:          float = 0.05,   # ARCH term  — weight on last squared innovation
    beta:           float = 0.94,   # GARCH term — weight on last conditional variance
    tick_size:      float = 0.0001,
    lam:            float = 0.0,    # jump intensity (jumps/year); 0 = no jumps
    mu_J:           float = 0.0,    # mean jump size in log-return units
    sigma_J:        float = 0.005,  # jump size std in log-return units (~0.5% per jump)
) -> tuple:
    """
    Simulate a GARCH(1,1) + Merton jump-diffusion price path.

    Variance process (GARCH drives diffusion only — jumps are additive):
        h[t+1] = ω + α·ε[t]² + β·h[t]
        ε[t]   = √h[t] · Z[t],   Z[t] ~ N(0,1) i.i.d.

    Jump process (Merton 1976):
        N[t]  ~ Bernoulli(λ·dt_year)   — 0 or 1 jump per step
        J[t]  ~ N(μ_J, σ_J²)           — jump size, fixed scale (not √dt)
        jump contribution: N[t] · J[t]

    Log-return at each step:
        r[t] = drift·dt_year − h[t]/2 + ε[t] + N[t]·J[t]

    The GARCH variance h is driven by ε only (diffusion), not by the jump.
    This is the standard GARCH-Jump separation: the vol process reflects
    clustering of normal moves; jumps are discrete, exogenous events.

    Parameters
    ----------
    alpha   : ARCH term. Typical FX intraday: 0.05.
    beta    : GARCH term. Typical FX intraday: 0.94. α+β must be < 1.
    lam     : Jump intensity in jumps/year. 252*3 ≈ 756 gives ~3 jumps/day.
              Set to 0 (default) to disable jumps.
    mu_J    : Mean log-return jump size. 0 = symmetric (no directional bias).
    sigma_J : Std of log-return jump size. 0.005 = ~0.5% price move per jump.

    Returns
    -------
    S            : np.ndarray (N+1,) — price path
    vol_realized : np.ndarray (N,)   — per-step diffusion innovations ε[t]
    h            : np.ndarray (N+1,) — conditional variance path (per-step units)
    dt           : float
    N            : int
    """
    if alpha + beta >= 1.0:
        raise ValueError(f"GARCH requires α+β < 1 for stationarity. Got {alpha+beta:.4f}.")

    dt      = time_grid[1] - time_grid[0]
    N       = len(time_grid) - 1
    dt_year = dt / TRADING_SECONDS_PER_YEAR

    sigma_step_sq = vol_annualized ** 2 * dt_year
    omega         = sigma_step_sq * (1.0 - alpha - beta)

    Z            = np.random.standard_normal(N)
    h            = np.empty(N + 1)
    h[0]         = sigma_step_sq
    vol_realized = np.empty(N)
    log_S        = np.empty(N + 1)
    log_S[0]     = np.log(S0)

    # ── Jump component ────────────────────────────────────────────────────────
    if lam > 0:
        jump_prob   = lam * dt_year                          # P(jump at step t)
        jump_occurs = np.random.binomial(1, jump_prob, N)   # 0/1 per step
        jump_sizes  = np.random.normal(mu_J, sigma_J, N)    # log-return jump size
        jumps       = jump_occurs * jump_sizes
    else:
        jumps = np.zeros(N)

    for t in range(N):
        eps_t           = np.sqrt(h[t]) * Z[t]
        vol_realized[t] = eps_t
        log_S[t + 1]    = log_S[t] + drift * dt_year - 0.5 * h[t] + eps_t + jumps[t]
        h[t + 1]        = omega + alpha * eps_t ** 2 + beta * h[t]  # jumps don't feed GARCH

    S = np.exp(log_S)
    if tick_size > 0:
        S = np.round(S / tick_size) * tick_size

    return S, vol_realized, h, dt, N
