import numpy as np
import matplotlib.pyplot as plt
from ..stock_simulation import Stock
from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR
from .spread_utils import compute_rv_zero_mean, evolve_s_excess


class Market(object):
    def __init__(self, stock: Stock):
        self.stock = stock
        self.vol_noise: np.array | None = None
        self.noised_mid_price: np.array | None = None
        self._time_grid: np.array | None = None
        self.ask_price_constant: np.array | None = None
        self.bid_price_constant: np.array | None = None
        self.ask_price_sto: np.array | None = None
        self.bid_price_sto: np.array | None = None
        self.ask_price_adaptive: np.array | None = None
        self.bid_price_adaptive: np.array | None = None
        self.ask_price_asym: np.array | None = None
        self.bid_price_asym: np.array | None = None
        # Realized vol arrays and window sizes stored for plotting
        self._rv_sto:      np.array | None = None
        self._rv_adaptive: np.array | None = None
        self._rv_asym:     np.array | None = None
        self._window_sto:      int | None = None
        self._window_adaptive: int | None = None
        self._window_asym:     int | None = None

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
        self.ask_price_constant = self.noised_mid_price + spread
        self.bid_price_constant = self.noised_mid_price - spread


    
    def build_stochastic_spread(self, window_size = 50, alpha = 0.016, tick_factor = 100):
        # Baseline half-spread: price-adaptive (full array, not just t=0)
        # tick_factor controls the minimum spread width in ticks
        spread_0 = self.noised_mid_price[0] * self.stock.tick_size * tick_factor / 2

        # Annualized realized volatility, rolling window of `window_size` steps
        vol_realized = self.stock.compute_realized_volatility(window_size=window_size)

        # Fill the warmup period with the stock's input vol (annualized) —
        # a market maker would have a prior on vol before the day starts
        vol_realized[:window_size] = self.stock.vol

        # Store for plotting — lets compare_spreads overlay vol on the width panel
        self._rv_sto     = vol_realized
        self._window_sto = window_size

        # Half-spread = baseline + volatility-scaled component
        # alpha is in price / annualized_vol units — tune to control vol sensitivity
        spread = np.round(spread_0 + alpha * vol_realized, 4)

        self.ask_price_sto = self.noised_mid_price + spread
        self.bid_price_sto = self.noised_mid_price - spread

    def build_adaptive_spread(self, window_size=50, alpha=0.016, tick_factor=100):
        # Baseline half-spread: same price-adaptive anchor as the stochastic spread
        spread_0 = self.noised_mid_price * self.stock.tick_size * tick_factor / 2

        # Annualized realized volatility, rolling window of `window_size` steps
        vol_realized = self.stock.compute_realized_volatility(window_size=window_size)

        # Warmup: use the known input vol as prior (no look-ahead)
        vol_realized[:window_size] = self.stock.vol

        # Vol excess: positive when market is more volatile than expected,
        # negative when calmer — this is what drives the spread adjustment
        vol_excess = vol_realized - self.stock.vol

        # Half-spread widens when vol_excess > 0, narrows when vol_excess < 0
        # alpha controls sensitivity (price / annualized_vol units)
        spread = np.round(spread_0 + alpha * vol_excess, 4)

        # Floor at 10% of baseline to prevent spread collapsing near zero
        spread = np.maximum(spread, spread_0 * 0.1)

        # Store for plotting
        self._rv_adaptive     = vol_realized
        self._window_adaptive = window_size

        self.ask_price_adaptive = self.noised_mid_price + spread
        self.bid_price_adaptive = self.noised_mid_price - spread

    def build_asymmetric_spread(
        self,
        s0:          float | None = None,
        alpha:       float | None = None,
        kappa_u:     float = 50.0,
        kappa_d:     float = 2.0,
        window_size: int   = 5000,
        tick_factor: int   = 100,
        sigma_s:     float = 0.0,
    ):
        """
        Build bid/ask prices using the asymmetric mean-reverting spread model.

        The spread S(t) = S_0 + S_excess(t) where:
          - S_0        is the static floor (minimum competitive spread, always present)
          - S_excess   mean-reverts toward a vol-driven target, but asymmetrically:
                       it widens fast (kappa_u) and tightens slowly (kappa_d)

        Parameters
        ----------
        s0          : static floor half-spread in price units.
                      Defaults to 1 tick (tick_size * 1).
        alpha       : spread sensitivity to vol (price / annualized vol).
                      Defaults to S_0 / (2 * sigma_ann) so that a 2-sigma vol event
                      doubles the spread — see doc section 12.
        kappa_u     : upward reversion speed in 1/s. Default 50 → half-life ~14ms.
        kappa_d     : downward reversion speed in 1/s. Default 2 → half-life ~350ms.
        window_size : lookback steps for the rolling realized vol estimate.
        sigma_s     : optional noise on the spread path (0 = disabled).
        """

        dt = self.stock.time_step
        print(f'dt : {dt} ')

        # ── Step 1-3: Rolling realized vol from the true GBM path ─────────────
        # Uses the zero-mean estimator (mean of r²) — more accurate than sample
        # std for short windows and zero-drift processes (see spread_utils.py)
        rv_ann = compute_rv_zero_mean(self.stock.simulation, window_size, dt)

        # ── Step 4: Static floor S_0 ───────────────────────────────────────────
        # Same formula as build_static_spread so all spreads share the same baseline:
        # half-spread = mid_price * tick_size * tick_factor / 2
        if s0 is None:
            s0 = self.noised_mid_price[0] * self.stock.tick_size * tick_factor / 2

        # ── Step 5: Alpha calibration ──────────────────────────────────────────
        # Default rule from doc section 12: a 2-sigma vol event doubles the spread
        # alpha = S_0 / (2 * sigma_ann), where sigma_ann is the input annual vol
        if alpha is None:
            alpha = s0 / (2 * self.stock.vol)

        # ── Step 6: Vol-driven target excess spread ────────────────────────────
        # S_star(t) = alpha * RV_ann(t)
        # When RV = input vol → S_star ≈ S_0/2, so total spread ≈ 1.5 * S_0
        # When RV = 2 * input vol → S_star ≈ S_0, so total spread ≈ 2 * S_0
        s_star = alpha * rv_ann

        # ── Step 7: Evolve S_excess forward (sequential, asymmetric) ──────────
        # kappa_u >> kappa_d: widening is fast (adverse selection fear),
        # tightening is slow (competitive pressure only after confirmed calm)
        s_excess = evolve_s_excess(s_star, kappa_u, kappa_d, dt, sigma_s)

        # ── Step 8: Full spread = floor + dynamic excess ───────────────────────
        # S_0 is the half-spread, so total bid-ask = 2 * (S_0 + S_excess)
        half_spread = s0 + s_excess

        # Store for plotting
        self._rv_asym     = rv_ann
        self._window_asym = window_size

        # ── Step 9: Bid and ask centered on the noised mid price ───────────────
        self.ask_price_asym = self.noised_mid_price + half_spread
        self.bid_price_asym = self.noised_mid_price - half_spread

    def build_spread(self, option = "Static", **kwargs):
        if option == "Static":
            self.build_static_spread(**kwargs)
        if option == "Sto":
            self.build_stochastic_spread(**kwargs)
        if option == "Adaptive":
            self.build_adaptive_spread(**kwargs)
        if option == "Asym":
            self.build_asymmetric_spread(**kwargs)



    # ── Plots ──────────────────────────────────────────────────────────────────

    def plot_noised_mid_price(self, series: list | None = None):
        """Plot the noised mid price with optional bid/ask overlays.

        Parameters
        ----------
        series : list of str, optional
            Which series to overlay. Accepted values: ``"static"``, ``"sto"``.
            Defaults to all available series.
        """
        if self.noised_mid_price is None:
            print("Noised mid price not computed — run generate_noised_mid_price first.")
            return

        valid = {"static", "sto", "adaptive", "asym"}
        if series is None:
            show = valid
        else:
            show = {s.lower() for s in series}
            unknown = show - valid
            if unknown:
                print(f"Warning: unknown series {unknown}. Valid options: {valid}")
            show &= valid

        t = self._time_grid / 3600

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        ax.plot(t, self.noised_mid_price, linewidth=0.8, color="#ff9500", label="Mid price")

        if "static" in show and self.ask_price_constant is not None and self.bid_price_constant is not None:
            ax.plot(t, self.ask_price_constant, linewidth=0.6, color="#ff4444", label="Ask (static)")
            ax.plot(t, self.bid_price_constant, linewidth=0.6, color="#44ff88", label="Bid (static)")
            ax.fill_between(t, self.bid_price_constant, self.ask_price_constant, color="#ff9500", alpha=0.10)

        if "sto" in show and self.ask_price_sto is not None and self.bid_price_sto is not None:
            ax.plot(t, self.ask_price_sto, linewidth=0.6, color="#4499ff", linestyle="--", label="Ask (sto)")
            ax.plot(t, self.bid_price_sto, linewidth=0.6, color="#ffcc00", linestyle="--", label="Bid (sto)")
            ax.fill_between(t, self.bid_price_sto, self.ask_price_sto, color="#4499ff", alpha=0.08)

        if "adaptive" in show and self.ask_price_adaptive is not None and self.bid_price_adaptive is not None:
            ax.plot(t, self.ask_price_adaptive, linewidth=0.6, color="#cc44ff", linestyle=":", label="Ask (adaptive)")
            ax.plot(t, self.bid_price_adaptive, linewidth=0.6, color="#ff44cc", linestyle=":", label="Bid (adaptive)")
            ax.fill_between(t, self.bid_price_adaptive, self.ask_price_adaptive, color="#cc44ff", alpha=0.08)

        if "asym" in show and self.ask_price_asym is not None and self.bid_price_asym is not None:
            ax.plot(t, self.ask_price_asym, linewidth=0.6, color="#ff8800", linestyle="-.", label="Ask (asym)")
            ax.plot(t, self.bid_price_asym, linewidth=0.6, color="#00ddff", linestyle="-.", label="Bid (asym)")
            ax.fill_between(t, self.bid_price_asym, self.ask_price_asym, color="#ff8800", alpha=0.06)

        if any(x is not None for x in [self.ask_price_constant, self.ask_price_sto, self.ask_price_adaptive, self.ask_price_asym]):
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

    def compare_spreads(self):
        """Ask panel, bid panel, then one width panel per generated spread."""
        if self.noised_mid_price is None:
            print("Noised mid price not computed — run generate_noised_mid_price first.")
            return

        has_static   = self.ask_price_constant is not None and self.bid_price_constant is not None
        has_sto      = self.ask_price_sto is not None and self.bid_price_sto is not None
        has_adaptive = self.ask_price_adaptive is not None and self.bid_price_adaptive is not None
        has_asym     = self.ask_price_asym is not None and self.bid_price_asym is not None

        if not has_static and not has_sto and not has_adaptive and not has_asym:
            print("No bid/ask series generated — run build_static_spread, build_stochastic_spread, build_adaptive_spread, or build_asymmetric_spread first.")
            return

        # 2 fixed panels (ask, bid) + one width panel per available spread
        n_width = sum([has_static, has_sto, has_adaptive, has_asym])
        n_rows  = 2 + n_width

        t = self._time_grid / 3600

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True)
        fig.patch.set_facecolor("#111111")

        ax_ask, ax_bid = axes[0], axes[1]
        width_axes = axes[2:]  # one per active spread, in order: static → sto → adaptive

        for ax in axes:
            ax.set_facecolor("#111111")
            ax.tick_params(colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#444444")
            ax.spines["bottom"].set_color("#444444")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#444444")

        for ax in (ax_ask, ax_bid):
            ax.plot(t, self.noised_mid_price, linewidth=0.7, color="#ff9500", alpha=0.5, label="Mid price")

        width_idx = 0

        def _style_twin(ax_right, color="#ffe033"):
            # Style the right-axis spine and ticks for the vol overlay
            ax_right.set_facecolor("#111111")
            ax_right.tick_params(colors=color)
            ax_right.spines["right"].set_color(color)
            ax_right.spines["top"].set_visible(False)
            ax_right.spines["left"].set_visible(False)
            ax_right.spines["bottom"].set_visible(False)

        def _corr_with_vol(width, rv):
            # Pearson correlation between spread width and realized vol
            # Both are (N+1,) arrays — correlation tells us how tightly the spread
            # tracks vol over time (1 = perfect, 0 = unrelated, -1 = inverse)
            return np.corrcoef(width, rv)[0, 1]

        def _add_vol_overlay(ax_w, width, rv, window, color):
            # Correlation between this spread's width and its vol input
            corr = _corr_with_vol(width, rv)

            # Append correlation to the existing title
            current_title = ax_w.get_title()
            ax_w.set_title(f"{current_title}    corr(width, RV) = {corr:.3f}",
                           color="white", fontsize=13, pad=10)

            # Second y-axis on the right for realized vol — different scale from width
            ax_vol = ax_w.twinx()
            _style_twin(ax_vol, color)
            ax_vol.plot(t, rv * 100, linewidth=0.7, color=color, alpha=0.7,
                        linestyle=":", label=f"RV ann. % (w={window})")
            ax_vol.set_ylabel("RV ann. (%)", color=color, fontsize=10)
            ax_vol.legend(loc="upper right", facecolor="#222222", edgecolor="#444444",
                          labelcolor="white", fontsize=9)

        if has_static:
            ax_ask.plot(t, self.ask_price_constant, linewidth=0.8, color="#ff4444", label="Ask (static)")
            ax_bid.plot(t, self.bid_price_constant, linewidth=0.8, color="#44ff88", label="Bid (static)")
            ax_w = width_axes[width_idx]
            ax_w.plot(t, self.ask_price_constant - self.bid_price_constant, linewidth=0.8, color="#ff4444",
                      label="Width (static)")
            ax_w.set_title("Spread width — Static (ask − bid)", color="white", fontsize=13, pad=10)
            ax_w.set_ylabel("Width", color="white", fontsize=12)
            ax_w.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=9)
            # Static spread has no vol input — no twin axis needed
            width_idx += 1

        if has_sto:
            ax_ask.plot(t, self.ask_price_sto, linewidth=0.8, color="#4499ff", linestyle="--", label="Ask (sto)")
            ax_bid.plot(t, self.bid_price_sto, linewidth=0.8, color="#ffcc00", linestyle="--", label="Bid (sto)")
            ax_w = width_axes[width_idx]
            ax_w.plot(t, self.ask_price_sto - self.bid_price_sto, linewidth=0.8, color="#4499ff",
                      linestyle="--", label="Width (sto)")
            ax_w.set_title("Spread width — Stochastic (ask − bid)", color="white", fontsize=13, pad=10)
            ax_w.set_ylabel("Width", color="white", fontsize=12)
            ax_w.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=9)
            if self._rv_sto is not None:
                _add_vol_overlay(ax_w, self.ask_price_sto - self.bid_price_sto, self._rv_sto, self._window_sto, "#ffe033")
            width_idx += 1

        if has_adaptive:
            ax_ask.plot(t, self.ask_price_adaptive, linewidth=0.8, color="#cc44ff", linestyle=":", label="Ask (adaptive)")
            ax_bid.plot(t, self.bid_price_adaptive, linewidth=0.8, color="#ff44cc", linestyle=":", label="Bid (adaptive)")
            ax_w = width_axes[width_idx]
            ax_w.plot(t, self.ask_price_adaptive - self.bid_price_adaptive, linewidth=0.8, color="#cc44ff",
                      linestyle=":", label="Width (adaptive)")
            ax_w.set_title("Spread width — Adaptive (ask − bid)", color="white", fontsize=13, pad=10)
            ax_w.set_ylabel("Width", color="white", fontsize=12)
            ax_w.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=9)
            if self._rv_adaptive is not None:
                _add_vol_overlay(ax_w, self.ask_price_adaptive - self.bid_price_adaptive, self._rv_adaptive, self._window_adaptive, "#ffe033")
            width_idx += 1

        if has_asym:
            ax_ask.plot(t, self.ask_price_asym, linewidth=0.8, color="#ff8800", linestyle="-.", label="Ask (asym)")
            ax_bid.plot(t, self.bid_price_asym, linewidth=0.8, color="#00ddff", linestyle="-.", label="Bid (asym)")
            ax_w = width_axes[width_idx]
            ax_w.plot(t, self.ask_price_asym - self.bid_price_asym, linewidth=0.8, color="#ff8800",
                      linestyle="-.", label="Width (asym)")
            ax_w.set_title("Spread width — Asymmetric (ask − bid)", color="white", fontsize=13, pad=10)
            ax_w.set_ylabel("Width", color="white", fontsize=12)
            ax_w.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=9)
            if self._rv_asym is not None:
                _add_vol_overlay(ax_w, self.ask_price_asym - self.bid_price_asym, self._rv_asym, self._window_asym, "#ffe033")

        ax_ask.set_title("Ask prices", color="white", fontsize=13, pad=10)
        ax_ask.set_ylabel("Price", color="white", fontsize=12)
        ax_ask.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=10)

        ax_bid.set_title("Bid prices", color="white", fontsize=13, pad=10)
        ax_bid.set_ylabel("Price", color="white", fontsize=12)
        ax_bid.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white", fontsize=10)

        axes[-1].set_xlabel("Time (hours)", color="white", fontsize=12)

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

    def sanity_check_spreads(self):
        """Print statistics for each generated bid/ask pair."""
        if self.noised_mid_price is None:
            raise ValueError("Run generate_noised_mid_price before calling sanity_check_spreads.")

        pairs = {}
        if self.ask_price_constant is not None and self.bid_price_constant is not None:
            pairs["Static"] = (self.ask_price_constant, self.bid_price_constant)
        if self.ask_price_sto is not None and self.bid_price_sto is not None:
            pairs["Stochastic"] = (self.ask_price_sto, self.bid_price_sto)
        if self.ask_price_adaptive is not None and self.bid_price_adaptive is not None:
            pairs["Adaptive"] = (self.ask_price_adaptive, self.bid_price_adaptive)
        if self.ask_price_asym is not None and self.bid_price_asym is not None:
            pairs["Asymmetric"] = (self.ask_price_asym, self.bid_price_asym)

        if not pairs:
            print("No bid/ask series generated yet — run build_static_spread, build_stochastic_spread, build_adaptive_spread, or build_asymmetric_spread first.")
            return

        mid = self.noised_mid_price
        dt_frac = self.stock.time_step / TRADING_SECONDS_PER_YEAR

        print("─" * 76)
        print(f"{'Metric':<44} " + "  ".join(f"{k:>14}" for k in pairs))
        print("─" * 76)

        spreads = {k: ask - bid for k, (ask, bid) in pairs.items()}
        rel_spreads = {k: spreads[k] / mid for k in pairs}
        ask_vols = {k: np.std(np.diff(np.log(ask))) * np.sqrt(1 / dt_frac) for k, (ask, _) in pairs.items()}
        bid_vols = {k: np.std(np.diff(np.log(bid))) * np.sqrt(1 / dt_frac) for k, (_, bid) in pairs.items()}
        ask_bias = {k: np.mean(ask - mid) for k, (ask, _) in pairs.items()}
        bid_bias = {k: np.mean(mid - bid) for k, (_, bid) in pairs.items()}

        def row(label, values):
            print(f"{label:<44} " + "  ".join(f"{v:>14.4f}" for v in values))

        def rowpct(label, values):
            print(f"{label:<44} " + "  ".join(f"{v:>13.4%}" for v in values))

        row("Spread mean (price units)",     [np.mean(spreads[k]) for k in pairs])
        row("Spread std  (price units)",     [np.std(spreads[k])  for k in pairs])
        row("Spread min  (price units)",     [np.min(spreads[k])  for k in pairs])
        row("Spread max  (price units)",     [np.max(spreads[k])  for k in pairs])
        rowpct("Relative spread mean (%)",   [np.mean(rel_spreads[k]) for k in pairs])
        rowpct("Relative spread std  (%)",   [np.std(rel_spreads[k])  for k in pairs])
        row("Ask bias vs mid (mean)",        [ask_bias[k] for k in pairs])
        row("Bid bias vs mid (mean)",        [bid_bias[k] for k in pairs])
        rowpct("Ask vol (ann., log-ret)",    [ask_vols[k] for k in pairs])
        rowpct("Bid vol (ann., log-ret)",    [bid_vols[k] for k in pairs])
        print("─" * 76)


