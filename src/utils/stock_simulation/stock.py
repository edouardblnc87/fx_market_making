import numpy as np
from scipy import stats
from .stochastic_simulation import *
from .config import TRADING_SECONDS_PER_YEAR
import matplotlib.pyplot as plt


class Stock(object):

    def __init__(self, drift: float, vol: float, origin: float = 100, tick_size: float = 0.0001):
        self.drift      = drift
        self.vol        = vol
        self.origin     = origin
        self.tick_size  = tick_size
        self.simulation: np.array | None = None   # shape (n_sims, N+1) after first run
        self._time_grid: np.array | None  = None   # seconds, set by simulate_gbm
        self.empty_sim = True
        self.sim_type: str | None = None           # 'gbm' or 'heston'

    def simulate_heston(self, n_days: int = 1, dt_seconds: float = 0.01,
                        v0=None, kappa: float = 2.0, theta=None,
                        xi: float = 0.3, rho: float = -0.1):
        """Simulate Heston stochastic-vol price path. Stores same attributes as
        simulate_gbm() plus variance_path, vol_path, and heston_params."""
        if v0 is None:
            v0 = self.vol ** 2
        if theta is None:
            theta = self.vol ** 2

        self.empty_sim = False
        time_grid = generate_time_grid(n_days, dt_seconds)
        S, v, vol_realized, dt, n_steps = generate_heston_path(
            time_grid, S0=self.origin, drift=self.drift,
            v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            tick_size=self.tick_size,
        )
        self.simulation    = S
        self._time_grid    = time_grid
        self.time_step     = dt
        self.n_steps       = n_steps
        self.vol_realized  = vol_realized
        self.variance_path = v
        self.vol_path      = np.sqrt(np.maximum(v, 0.0))
        self.heston_params = dict(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
        self.sim_type      = 'heston'

    def simulate_gbm(self, n_days: int = 30, dt_seconds: float = 0.05):

        self.empty_sim = False
        time_grid = generate_time_grid(n_days, dt_seconds)
        path, vol_realized, mu_realized, dt, n_steps  = generate_gbm_path(time_grid, self.origin, self.drift, self.vol, self.tick_size)
        self.vol_realized = vol_realized
        self.mu_realized = mu_realized
        self.time_step = dt
        self.n_steps = n_steps
        self._time_grid = time_grid

        self.simulation =  path
        self.sim_type   = 'gbm'
    
    def inject_vol_shock(self, t_start_frac: float = 0.4, duration_s: float = 300.0,
                         shock_factor: float = 3.0):
        """Splice a vol shock into the existing price path.

        Re-draws the log-return increments for the shock window using
        shock_factor * vol, then reconnects the original increments (at normal
        vol) from the shock end onwards.  Everything before t_start is untouched.

        Parameters
        ----------
        t_start_frac : float
            Start of shock as a fraction of the total path (0 to 1).
        duration_s   : float
            Duration of the shock in seconds.
        shock_factor : float
            Multiplier on self.vol during the shock (e.g. 3.0 = triple vol).
        """
        if self.simulation is None:
            print("No path generated yet.")
            return

        dt      = self.time_step
        N       = self.n_steps
        dt_year = dt / TRADING_SECONDS_PER_YEAR

        t0          = int(t_start_frac * N)
        shock_steps = min(round(duration_s / dt), N - t0)
        t_end       = t0 + shock_steps

        # ── Shock-period increments ───────────────────────────────────────────
        vol_shock = shock_factor * self.vol
        sigma_dt  = vol_shock * np.sqrt(dt_year)
        mu_dt     = (self.drift - 0.5 * vol_shock ** 2) * dt_year

        Z_shock        = np.random.standard_normal(shock_steps)
        shock_log_rets = mu_dt + sigma_dt * Z_shock

        # ── Rebuild log-price path ────────────────────────────────────────────
        orig_log_S = np.log(self.simulation)
        log_S      = orig_log_S.copy()

        # shock window
        log_S[t0 + 1 : t_end + 1] = log_S[t0] + np.cumsum(shock_log_rets)

        # post-shock: keep original increments, shifted to new level at t_end
        orig_post_rets = np.diff(orig_log_S[t_end:])
        if len(orig_post_rets) > 0:
            log_S[t_end + 1 :] = log_S[t_end] + np.cumsum(orig_post_rets)

        S = np.exp(log_S)
        if self.tick_size > 0:
            S = np.round(S / self.tick_size) * self.tick_size

        self.simulation  = S
        self.vol_realized = np.diff(log_S)   # update residuals to match new path
        self.shock_params = dict(
            t0=t0, t_end=t_end, t_start_frac=t_start_frac,
            duration_s=duration_s, shock_factor=shock_factor,
        )

    #return the series of annualized realized vol over a window n
    def compute_realized_volatility(self, window_size):
        if self.simulation is None:
            print("No path generated yet — run simulate_gbm first.")
            return
        else:
            coeff_annualization = np.sqrt(TRADING_SECONDS_PER_DAY * 252/self.time_step)
            print(f"Coefficient annualization : {coeff_annualization}")
            log_returns = np.diff(np.log(self.simulation))
            vol = np.zeros(len(self.simulation))
            for t in range(window_size, len(self.simulation)):                                                                                                                                                                       
                vol[t] = np.std(log_returns[t - window_size : t])                                                                                                                                                              
            return vol * coeff_annualization
             


# ── Plots ──────────────────────────────────────────────────────────────────

    def plot_path(self):
        if self.simulation is None:
            print("No path generated yet — run simulate_gbm or simulate_heston first.")
            return

        t = self._time_grid / 3600   # seconds → hours, more readable on x-axis

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        ax.plot(t, self.simulation, linewidth=0.8, color="#00bfff")

        if self.sim_type == 'heston':
            p = self.heston_params
            title = (f"Heston — S₀={self.origin}  σ={self.vol:.0%}  μ={self.drift:.0%}"
                     f"  κ={p['kappa']}  θ={p['theta']:.4f}  ξ={p['xi']}  ρ={p['rho']}")
        else:
            title = f"GBM — S₀={self.origin}  σ={self.vol:.0%}  μ={self.drift:.0%}"

        ax.set_title(title, color="white", fontsize=13, pad=12)
        ax.set_xlabel("Time (hours)", color="white", fontsize=12)
        ax.set_ylabel("Price",        color="white", fontsize=12)

        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#444444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")

        plt.tight_layout()
        plt.show()

    def plot_vol_path(self):
        """2-panel dark figure: price path (cyan) + instantaneous vol path (magenta)."""
        if not hasattr(self, 'vol_path') or self.vol_path is None:
            print("No Heston path generated yet — run simulate_heston first.")
            return

        t = self._time_grid / 3600

        fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.patch.set_facecolor("#111111")

        for ax in (ax_price, ax_vol):
            ax.set_facecolor("#111111")
            ax.tick_params(colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#444444")
            ax.spines["bottom"].set_color("#444444")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#444444")

        ax_price.plot(t, self.simulation, linewidth=0.7, color="#00bfff")
        ax_price.set_title(
            f"Heston — S₀={self.origin}  σ={self.vol:.0%}  ξ={self.heston_params['xi']}  ρ={self.heston_params['rho']}",
            color="white", fontsize=13, pad=10,
        )
        ax_price.set_ylabel("Price", color="white", fontsize=11)

        # vol_path = sqrt(v), v is already in annualized variance units
        ax_vol.plot(t, self.vol_path, linewidth=0.7, color="#ff44cc")
        ax_vol.set_title("Instantaneous Volatility (ann.)", color="white", fontsize=13, pad=10)
        ax_vol.set_xlabel("Time (hours)", color="white", fontsize=11)
        ax_vol.set_ylabel("Vol (ann.)",   color="white", fontsize=11)

        plt.tight_layout()
        plt.show()

    def sanity_check(self):
        if self.simulation is None:
            print("No path generated yet — run simulate_gbm or simulate_heston first.")
            return

        log_rets  = np.diff(np.log(self.simulation))
        dt_frac   = self.time_step / TRADING_SECONDS_PER_YEAR

        vol_ann   = np.std(log_rets) * np.sqrt(1.0 / dt_frac)
        drift_ann = np.mean(log_rets) / dt_frac
        skewness  = stats.skew(log_rets)
        kurt      = stats.kurtosis(log_rets)          # excess kurtosis

        expected_drift = self.drift - 0.5 * self.vol ** 2

        is_heston = self.sim_type == 'heston'
        model_tag = "Heston" if is_heston else "GBM"

        # For Heston: fat tails only emerge at coarser aggregation (kurtosis ~ ξ²/κ²τ → 0 as dt→0).
        # Aggregate into ~60s buckets so the variance has time to drift between observations.
        if is_heston:
            agg_steps = max(1, round(60.0 / self.time_step))
            n_buckets = len(log_rets) // agg_steps
            agg_rets  = log_rets[: n_buckets * agg_steps].reshape(n_buckets, agg_steps).sum(axis=1)
            kurt_label = f"Excess kurtosis (60s agg.)"
            kurt_display = stats.kurtosis(agg_rets)
        else:
            kurt_label   = "Excess kurtosis"
            kurt_display = kurt

        print("─" * 62)
        print(f"  Sanity check — {model_tag}")
        print("─" * 62)
        print(f"{'Metric':<35} {'Value':>12}  {'Expected':>12}")
        print("─" * 62)
        print(f"{'Realized ann. vol':<35} {vol_ann:>11.4%}  {self.vol:>11.4%}")
        print(f"{'Realized ann. drift (Itô)':<35} {drift_ann:>12.4f}  {expected_drift:>12.4f}")
        print(f"{'Skewness of log-returns':<35} {skewness:>12.4f}  {'~0':>12}")
        kurt_expected = '>0 (fat tails)' if is_heston else '~0'
        print(f"{kurt_label:<35} {kurt_display:>12.4f}  {kurt_expected:>12}")
        print(f"{'Min price':<35} {self.simulation.min():>12.4f}  {'—':>12}")
        print(f"{'Max price':<35} {self.simulation.max():>12.4f}  {'—':>12}")
        print(f"{'Final price':<35} {self.simulation[-1]:>12.4f}  {'—':>12}")

        # ── Heston-specific rows ───────────────────────────────────────────────
        if hasattr(self, 'variance_path') and self.variance_path is not None:
            p = self.heston_params
            # Use 60s aggregated returns for autocorr too — same reason as kurtosis
            sq_agg  = agg_rets ** 2
            autocorr = np.corrcoef(sq_agg[:-1], sq_agg[1:])[0, 1]
            feller    = 2.0 * p['kappa'] * p['theta'] / (p['xi'] ** 2)
            xi_realized = np.std(np.diff(self.variance_path) /
                                 np.sqrt(np.maximum(self.variance_path[:-1], 1e-10) *
                                         (self.time_step / TRADING_SECONDS_PER_YEAR)))
            print("─" * 62)
            print(f"{'Vol of vol realized':<35} {xi_realized:>11.4f}  {p['xi']:>11.4f}")
            print(f"{'Autocorr sq. returns (60s agg.)':<35} {autocorr:>12.4f}  {'>0 expected':>12}")
            print(f"{'Feller condition (2κθ/ξ²)':<35} {feller:>12.4f}  {'>1 required':>12}")

        print("─" * 62)

        # ── Plot: price path + log-return histogram ────────────────────────────
        t = self._time_grid / 3600

        fig, (ax_path, ax_hist) = plt.subplots(1, 2, figsize=(16, 5))
        fig.patch.set_facecolor("#111111")

        for ax in (ax_path, ax_hist):
            ax.set_facecolor("#111111")
            ax.tick_params(colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#444444")
            ax.spines["bottom"].set_color("#444444")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#444444")

        # Left: price path
        ax_path.plot(t, self.simulation, linewidth=0.8, color="#00bfff")
        ax_path.set_title(f"{model_tag} Price Path", color="white", fontsize=13)
        ax_path.set_xlabel("Time (hours)", color="white", fontsize=11)
        ax_path.set_ylabel("Price",        color="white", fontsize=11)

        # Right: log-return histogram with fitted normal overlay
        ax_hist.hist(log_rets, bins=80, density=True, color="#00bfff", alpha=0.6, label="Log returns")

        mu_fit, sigma_fit = np.mean(log_rets), np.std(log_rets)
        x = np.linspace(log_rets.min(), log_rets.max(), 400)
        ax_hist.plot(x, stats.norm.pdf(x, mu_fit, sigma_fit),
                     color="#ff9500", linewidth=1.5, label="N(μ, σ²) fit")

        ax_hist.set_title("Log-Return Distribution", color="white", fontsize=13)
        ax_hist.set_xlabel("Log return",  color="white", fontsize=11)
        ax_hist.set_ylabel("Density",     color="white", fontsize=11)
        ax_hist.legend(facecolor="#222222", edgecolor="#444444",
                       labelcolor="white", fontsize=10)

        fig.suptitle(
            f"Sanity check [{model_tag}] — σ={self.vol:.0%}  μ={self.drift:.0%}  dt={self.time_step:.3f}s",
            color="white", fontsize=14, y=1.02,
        )
        plt.tight_layout()
        plt.show()
