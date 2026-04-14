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

# ── Plots ──────────────────────────────────────────────────────────────────

    def plot_path(self):
        if self.simulation is None:
            print("No path generated yet — run simulate_gbm first.")
            return

        t = self._time_grid / 3600   # seconds → hours, more readable on x-axis

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")


        
        ax.plot(t, self.simulation, linewidth=0.8, color="#00bfff")

        ax.set_title(
            f"GBM — S₀={self.origin}  σ={self.vol:.0%}  μ={self.drift:.0%}",
            color="white", fontsize=14, pad=12,
        )
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

    def sanity_check(self):
        if self.simulation is None:
            print("No path generated yet — run simulate_gbm first.")
            return

        log_rets  = np.diff(np.log(self.simulation))
        dt_frac   = self.time_step / TRADING_SECONDS_PER_YEAR

        vol_ann   = np.std(log_rets) * np.sqrt(1.0 / dt_frac)
        drift_ann = np.mean(log_rets) / dt_frac
        skewness  = stats.skew(log_rets)
        kurt      = stats.kurtosis(log_rets)          # excess kurtosis

        expected_drift = self.drift - 0.5 * self.vol ** 2

        print("─" * 62)
        print(f"{'Metric':<35} {'Value':>12}  {'Expected':>12}")
        print("─" * 62)
        print(f"{'Realized ann. vol':<35} {vol_ann:>11.4%}  {self.vol:>11.4%}")
        print(f"{'Realized ann. drift (Itô)':<35} {drift_ann:>12.4f}  {expected_drift:>12.4f}")
        print(f"{'Skewness of log-returns':<35} {skewness:>12.4f}  {'~0':>12}")
        print(f"{'Excess kurtosis':<35} {kurt:>12.4f}  {'~0':>12}")
        print(f"{'Min price':<35} {self.simulation.min():>12.4f}  {'—':>12}")
        print(f"{'Max price':<35} {self.simulation.max():>12.4f}  {'—':>12}")
        print(f"{'Final price':<35} {self.simulation[-1]:>12.4f}  {'—':>12}")
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
        ax_path.set_title("GBM Price Path", color="white", fontsize=13)
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
            f"Sanity check — σ={self.vol:.0%}  μ={self.drift:.0%}  dt={self.time_step:.3f}s",
            color="white", fontsize=14, y=1.02,
        )
        plt.tight_layout()
        plt.show()
