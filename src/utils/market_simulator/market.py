import numpy as np
import matplotlib.pyplot as plt
from ..stock_simulation import Stock
from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR


class Market(object):
    def __init__(self, stock: Stock):
        self.stock = stock
        self.vol_noise: np.array | None = None
        self.noised_mid_price: np.array | None = None
        self._time_grid: np.array | None = None
        self.ask_price: np.array | None = None
        self.bid_price: np.array | None = None

    def generate_noise(self, vol_factor: float):
        dt_frac = self.stock.time_step / TRADING_SECONDS_PER_YEAR
        Z = np.random.standard_normal(self.stock.n_steps + 1)
        self.vol_noise = vol_factor * self.stock.vol * np.sqrt(dt_frac) * Z
        return self.vol_noise

    def generate_noised_mid_price(self, vol_factor: float = 0.1):
        if self.stock.empty_sim:
            print("Stock path hasn't been initialized — running simulation.")
            self.stock.simulate_gbm()
            self.stock.plot_path()

        self.generate_noise(vol_factor)
        self._time_grid = self.stock._time_grid
        self.noised_mid_price = self.stock.simulation + self.vol_noise

    def build_static_spread(self, tick_factor = 100):
        if self.noised_mid_price is None:
            print(f"Error, no prices generated for this market, run generate_noised_mid_price")
        spread = self.noised_mid_price[0] * self.stock.tick_size * tick_factor/2
        self.ask_price = self.noised_mid_price + spread
        self.bid_price = self.noised_mid_price - spread

    def build_spread(self, option = "Static"):
        if option == "Static":
            self.build_static_spread()

    # ── Plots ──────────────────────────────────────────────────────────────────

    def plot_noised_mid_price(self):
        if self.noised_mid_price is None:
            print("Noised mid price not computed — run generate_noised_mid_price first.")
            return

        t = self._time_grid / 3600

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        ax.plot(t, self.noised_mid_price, linewidth=0.8, color="#ff9500", label="Mid price")

        if self.ask_price is not None and self.bid_price is not None:
            ax.plot(t, self.ask_price, linewidth=0.6, color="#ff4444", label="Ask")
            ax.plot(t, self.bid_price, linewidth=0.6, color="#44ff88", label="Bid")
            ax.fill_between(t, self.bid_price, self.ask_price, color="#ff9500", alpha=0.15)
            ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=10)

        ax.set_title("Noised Mid Price", color="white", fontsize=14, pad=12)
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

    def plot_comparison(self):
        if self.noised_mid_price is None or self.stock.simulation is None:
            raise ValueError("Both GBM path and noised mid price must be computed first.")

        t         = self._time_grid / 3600
        deviation = self.noised_mid_price - self.stock.simulation
        dev_mean  = np.mean(deviation)
        dev_std   = np.std(deviation)

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig.patch.set_facecolor("#111111")

        for ax in (ax_top, ax_bot):
            ax.set_facecolor("#111111")
            ax.tick_params(colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#444444")
            ax.spines["bottom"].set_color("#444444")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#444444")

        # Top: both paths
        ax_top.plot(t, self.stock.simulation,  linewidth=0.8, color="#00bfff", label="True GBM")
        ax_top.plot(t, self.noised_mid_price,   linewidth=0.8, color="#ff9500", label="Noised mid")
        ax_top.set_title("True GBM vs Noised Mid Price", color="white", fontsize=14, pad=12)
        ax_top.set_ylabel("Price", color="white", fontsize=12)
        ax_top.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=10)

        # Bottom: deviation
        ax_bot.fill_between(t, deviation, 0, color="#ff9500", label="Deviation")
        ax_bot.axhline(0, color="#444444", linewidth=0.8)
        ax_bot.set_title(
            f"Deviation (noised − true)   mean={dev_mean:.4f}  std={dev_std:.4f}",
            color="white", fontsize=13, pad=10,
        )
        ax_bot.set_xlabel("Time (hours)", color="white", fontsize=12)
        ax_bot.set_ylabel("Price diff",   color="white", fontsize=12)
        ax_bot.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=10)

        plt.tight_layout()
        plt.show()

    # ── Sanity check ──────────────────────────────────────────────────────────

    def sanity_check(self):
        if self.noised_mid_price is None or self.stock.simulation is None:
            raise ValueError("Run generate_noised_mid_price before calling sanity_check.")

        noise         = self.noised_mid_price - self.stock.simulation
        dt_frac       = self.stock.time_step / TRADING_SECONDS_PER_YEAR

        noise_std_price = np.std(noise)
        approx_ref      = (np.std(noise) / np.std(self.stock.simulation)
                           * self.stock.vol * np.mean(self.stock.simulation))

        noise_increments = np.diff(noise)
        noise_vol_ann    = np.std(noise_increments) * np.sqrt(1.0 / dt_frac)

        log_rets_noised = np.diff(np.log(self.noised_mid_price))
        total_vol_ann   = np.std(log_rets_noised) * np.sqrt(1.0 / dt_frac)

        gbm_vol_ann     = np.std(self.stock.vol_realized) * np.sqrt(1.0 / dt_frac)

        true_increments = np.diff(self.stock.simulation)
        nsr             = np.std(noise) / np.std(true_increments)

        print("─" * 68)
        print(f"{'Metric':<40} {'Value':>14}  {'Reference':>12}")
        print("─" * 68)
        print(f"{'Realized noise std (price units)':<40} {noise_std_price:>14.4f}  {'—':>12}")
        print(f"{'Realized noise vol (ann. %)':<40} {noise_vol_ann:>13.4%}  {'—':>12}")
        print(f"{'Total mid price vol (ann. %)':<40} {total_vol_ann:>13.4%}  {self.stock.vol:>11.4%}≤")
        print(f"{'True GBM vol (ann. %)':<40} {gbm_vol_ann:>13.4%}  {self.stock.vol:>11.4%}")
        print(f"{'Noise-to-signal ratio':<40} {nsr:>14.4f}  {'—':>12}")
        print("─" * 68)


