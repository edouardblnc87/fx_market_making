"""
diagnostics.py — Visual analysis of a completed market-making simulation.

Answers four questions at a glance:
  1. What happened? (price path, fills, hedges, inventory)
  2. Is the spread right? (spread level, skew, premium over reference)
  3. Is the order flow as expected? (fill rate, imbalance, level utilisation)
  4. Should I change any parameters? (scorecard with colour-coded recommendations)

Usage
-----
    from utils.report.diagnostics import DiagnosticsReport
    dr = DiagnosticsReport(ctrl)
    dr.report()          # all figures
    dr.plot_overview()   # individual figures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────────────────
_BG   = "#0d1117"
_AX   = "#161b22"
_GRID = "#21262d"
_TXT  = "#e6edf3"
_MID  = "#58a6ff"
_BID  = "#3fb950"
_ASK  = "#f85149"
_INV  = "#d2a8ff"
_HED  = "#f0883e"
_SPR  = "#ffa657"
_REF  = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _AX,
    "axes.edgecolor": _GRID, "axes.labelcolor": _TXT,
    "xtick.color": _TXT, "ytick.color": _TXT,
    "text.color": _TXT, "grid.color": _GRID,
    "grid.linewidth": 0.5, "axes.grid": True,
    "legend.facecolor": _AX, "legend.edgecolor": _GRID,
    "legend.labelcolor": _TXT, "font.size": 9,
})

_MAX_PTS = 3_000   # downsample cap for step_log plots


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ds(df: pd.DataFrame, n: int = _MAX_PTS) -> pd.DataFrame:
    """Downsample to at most n rows."""
    stride = max(1, len(df) // n)
    return df.iloc[::stride]


def _vlines(ax, xs, color=_HED, alpha=0.7, lw=1.2, label=None):
    for i, x in enumerate(xs):
        ax.axvline(x, color=color, alpha=alpha, lw=lw,
                   linestyle="--", label=label if i == 0 else None)


def _fig(title: str, figsize=(14, 8)):
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    fig.suptitle(title, color=_TXT, fontsize=11, y=0.98)
    return fig


def _ax_style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(_AX)
    ax.tick_params(colors=_TXT)
    if xlabel:
        ax.set_xlabel(xlabel, color=_TXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=_TXT)
    if title:
        ax.set_title(title, color=_TXT, fontsize=9)


# ── Main class ─────────────────────────────────────────────────────────────────

class DiagnosticsReport:
    """
    Visual diagnostics for a completed Controller simulation.

    Parameters
    ----------
    ctrl    : Controller after simulate() has been called
    n_days  : trading days simulated (used for per-day normalisation)
    """

    def __init__(self, ctrl, n_days: float | None = None):
        self.ctrl = ctrl
        self.mm   = ctrl.quoter
        self.cfg  = ctrl.quoter.cfg
        self.K    = ctrl.quoter.capital_K

        sl = ctrl.step_log
        th = ctrl.trade_history

        self.sl  = sl
        self.th  = th
        self.mm_fills = th[~th["is_hedge"]].copy()
        self.hedges   = th[th["is_hedge"]].copy()

        # time axis: use 't' column if it looks like seconds, else use step index
        if "t" in sl.columns and sl["t"].max() > 1:
            self.t = sl["t"].values
        else:
            self.t = sl.index.values

        # spread in bps
        self.spA = ((sl["ask_A"] - sl["bid_A"]) / sl["mid_A"] * 1e4).values
        self.spB = ((sl["ask_B"] - sl["bid_B"]) / sl["mid_B"] * 1e4).values

        # hedge trigger level (EUR notional)
        self.limit = self.cfg.delta_limit * self.K * 0.5

        # hedge event times
        self.hedge_steps = self.hedges["step"].unique() if len(self.hedges) else np.array([])
        self.hedge_t = (
            sl.loc[sl["step"].isin(self.hedge_steps), "t"].values
            if "t" in sl.columns and len(self.hedge_steps)
            else self.hedge_steps.astype(float)
        )

        # n_days guess
        if n_days is not None:
            self.n_days = n_days
        else:
            dt = ctrl.market_B.stock.time_step
            self.n_days = max(1.0, len(sl) * dt / 86400)

    # ── Figure 1: Overview ─────────────────────────────────────────────────────

    def plot_overview(self):
        """Price path + fills + hedges | Inventory | Spread."""
        sl  = _ds(self.sl)
        t   = sl["t"].values if "t" in sl.columns else sl.index.values

        fig = _fig("Overview — Price, Inventory, Spread", figsize=(14, 9))
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35,
                                height_ratios=[2.2, 1.5, 1.3])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        # ── Panel 1: Price ──────────────────────────────────────────────────
        ax1.plot(t, sl["fair_mid"], color=_MID, lw=0.8, label="Fair mid")
        ax1.fill_between(t, sl["bid_A"], sl["ask_A"],
                         alpha=0.15, color=_MID, label="Quoted A spread")

        # fill events
        mf = self.mm_fills
        if len(mf):
            t_fill = mf["t"].values if "t" in mf.columns else mf["step"].values
            buys  = mf[mf["direction"] == "buy"]
            sells = mf[mf["direction"] == "sell"]
            t_buy  = buys["t"].values  if "t" in buys.columns  else buys["step"].values
            t_sell = sells["t"].values if "t" in sells.columns else sells["step"].values
            ax1.scatter(t_buy,  buys["price"],  marker="^", s=18,
                        color=_BID, zorder=5, label=f"Client buy fills ({len(buys):,})")
            ax1.scatter(t_sell, sells["price"], marker="v", s=18,
                        color=_ASK, zorder=5, label=f"Client sell fills ({len(sells):,})")

        # hedge markers
        _vlines(ax1, self.hedge_t, label=f"Hedge ({len(self.hedge_steps)} events)")
        _ax_style(ax1, ylabel="Price", title="Price path  •  fills  •  hedges")
        ax1.legend(loc="upper left", fontsize=7.5, ncol=3)

        # ── Panel 2: Inventory ──────────────────────────────────────────────
        inv = sl["inventory"].values
        inv_usd = inv * sl["fair_mid"].values
        ax2.plot(t, inv / 1e3, color=_INV, lw=0.9, label="Inventory (kEUR)")
        ax2.axhline( self.limit / 1e3, color=_ASK, lw=1, ls="--",
                    label=f"Hedge trigger ±{self.limit/1e3:.0f} kEUR")
        ax2.axhline(-self.limit / 1e3, color=_ASK, lw=1, ls="--")
        ax2.axhline(0, color=_REF, lw=0.6)
        _vlines(ax2, self.hedge_t)
        _ax_style(ax2, ylabel="kEUR", title="Inventory (EUR) with hedge trigger")
        ax2.legend(loc="upper left", fontsize=7.5)

        # ── Panel 3: Spread ─────────────────────────────────────────────────
        ax3.plot(t, self.spA[::max(1, len(self.sl) // _MAX_PTS)],
                 color=_SPR, lw=0.7, label="Spread A (bps)")
        ax3.plot(t, self.spB[::max(1, len(self.sl) // _MAX_PTS)],
                 color=_REF, lw=0.6, alpha=0.7, label="Spread B (bps)")
        _vlines(ax3, self.hedge_t)
        _ax_style(ax3, xlabel="Time (s)", ylabel="bps",
                  title="Quoted spread A vs reference B")
        ax3.legend(loc="upper right", fontsize=7.5)

        plt.tight_layout()
        plt.show()

    # ── Figure 2: Spread & Quote Legs ─────────────────────────────────────────

    def plot_spread_and_skew(self):
        """
        Three panels:
          1. Spread A vs B over time, with realised vol on second axis
             → does the spread track vol? is gamma calibrated?
          2. Bid and ask offsets from fair_mid over time
             → shows asymmetric leg shifting driven by inventory skew
          3. Inventory coloured bar alongside bid/ask offset difference
             → links skew magnitude directly to inventory level
        """
        sl  = _ds(self.sl)
        t   = sl["t"].values if "t" in sl.columns else sl.index.values

        fair  = sl["fair_mid"].values
        bid_A = sl["bid_A"].values
        ask_A = sl["ask_A"].values
        inv   = sl["inventory"].values

        # half-spread offsets from fair mid in bps
        ask_offset = np.where(fair > 0, (ask_A - fair) / fair * 1e4, np.nan)
        bid_offset = np.where(fair > 0, (fair - bid_A) / fair * 1e4, np.nan)
        # positive = leg is above its neutral position
        # ask_offset > bid_offset → ask pushed up (selling pressure = long inventory)
        # bid_offset > ask_offset → bid pushed up (buying pressure = short inventory)
        leg_diff = ask_offset - bid_offset   # >0: ask wider, <0: bid wider

        spA = (ask_A - bid_A) / fair * 1e4
        spB = (sl["ask_B"] - sl["bid_B"]) / sl["mid_B"] * 1e4

        # rolling realised vol (annualised, bps)
        mid_B_full  = self.sl["mid_B"].values
        log_ret     = np.diff(np.log(np.where(mid_B_full > 0, mid_B_full, np.nan)))
        dt          = self.ctrl.market_B.stock.time_step
        from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR
        win         = max(1, int(600 / dt))   # 10-min rolling window
        vol_roll    = pd.Series(log_ret ** 2).rolling(win, min_periods=1).mean() ** 0.5
        vol_ann_bps = (vol_roll / np.sqrt(dt / TRADING_SECONDS_PER_YEAR) * 1e4).values
        # align length (diff drops one point)
        vol_ann_bps = np.append(vol_ann_bps, vol_ann_bps[-1])
        stride      = max(1, len(self.sl) // _MAX_PTS)
        vol_ds      = vol_ann_bps[::stride][:len(t)]

        fig = _fig("Spread Dynamics & Quote-Leg Asymmetry", figsize=(14, 9))
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.42)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        # ── Panel 1: Spread A vs B + realised vol ───────────────────────────
        ax1.plot(t, spA, color=_SPR, lw=0.7, label=f"Spread A  (mean {spA.mean():.2f} bps)")
        ax1.plot(t, spB, color=_REF, lw=0.6, alpha=0.8,
                 label=f"Spread B  (mean {spB.mean():.2f} bps)")
        ax1.fill_between(t, spB, spA, alpha=0.18, color=_SPR)
        _vlines(ax1, self.hedge_t)

        ax1b = ax1.twinx()
        ax1b.plot(t, vol_ds, color=_MID, lw=0.6, alpha=0.6, label="Realised vol (ann. bps)")
        ax1b.set_ylabel("Realised vol (ann. bps)", color=_MID, fontsize=8)
        ax1b.tick_params(colors=_MID)
        ax1b.set_facecolor(_AX)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper right", ncol=2)
        _ax_style(ax1, ylabel="bps",
                  title="Spread A vs B  +  realised vol  —  does spread track vol? (it should)")

        # ── Panel 2: Ask offset and bid offset from fair_mid ────────────────
        ax2.plot(t, ask_offset, color=_ASK, lw=0.7, label="Ask offset from fair (bps)")
        ax2.plot(t, bid_offset, color=_BID, lw=0.7, label="Bid offset from fair (bps)")
        ax2.fill_between(t, bid_offset, ask_offset, alpha=0.1, color=_SPR)
        ax2.axhline(float(np.nanmean(ask_offset)), color=_ASK, lw=1, ls=":",
                    label=f"mean ask = {np.nanmean(ask_offset):.2f} bps")
        ax2.axhline(float(np.nanmean(bid_offset)), color=_BID, lw=1, ls=":",
                    label=f"mean bid = {np.nanmean(bid_offset):.2f} bps")
        _vlines(ax2, self.hedge_t)
        _ax_style(ax2, ylabel="bps from fair_mid",
                  title="Ask & bid leg distance from fair_mid  —  symmetric = no skew, diverging = inventory pressure")
        ax2.legend(fontsize=7.5, ncol=2)

        # ── Panel 3: Leg difference vs inventory ────────────────────────────
        ax3.plot(t, leg_diff, color=_SPR, lw=0.7,
                 label="ask_offset − bid_offset  (>0 = ask wider = long inventory)")
        ax3.axhline(0, color=_REF, lw=0.8)

        ax3b = ax3.twinx()
        ax3b.plot(t, inv / 1e3, color=_INV, lw=0.6, alpha=0.5, label="Inventory (kEUR)")
        ax3b.axhline(0, color=_REF, lw=0.4)
        ax3b.set_ylabel("Inventory (kEUR)", color=_INV, fontsize=8)
        ax3b.tick_params(colors=_INV)
        ax3b.set_facecolor(_AX)
        _vlines(ax3, self.hedge_t)

        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax3b.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, fontsize=7.5, loc="upper left", ncol=2)
        _ax_style(ax3, xlabel="Time (s)", ylabel="bps",
                  title="Quote-leg asymmetry vs inventory  —  skew should track inventory sign")

    # ── Figure 3: Fill Rate & Order Flow ──────────────────────────────────────

    def plot_fills(self):
        """Fill rate, direction imbalance, level utilisation."""
        mf  = self.mm_fills
        sl  = self.sl
        dt  = self.ctrl.market_B.stock.time_step

        fig = _fig("Fill Rate & Order Flow", figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2])

        # ── Rolling fill rate ───────────────────────────────────────────────
        window_steps = max(1, int(600 / dt))   # 10-min window
        fill_rate = sl["fills_this_step"].rolling(window_steps, min_periods=1).mean() / dt
        t = sl["t"].values if "t" in sl.columns else sl.index.values
        _sl_ds = _ds(sl)
        t_ds   = _sl_ds["t"].values if "t" in _sl_ds.columns else _sl_ds.index.values
        fr_ds  = fill_rate.iloc[::max(1, len(sl) // _MAX_PTS)]

        ax1.plot(t_ds, fr_ds, color=_SPR, lw=0.8, label="Rolling fill rate (fills/s)")
        _vlines(ax1, self.hedge_t)

        # theoretical lambda
        cfg = self.cfg
        A_buy  = getattr(cfg, 'A_buy',  0.007)
        A_sell = getattr(cfg, 'A_sell', 0.007)
        k_buy  = getattr(cfg, 'k_buy',  0.3)
        k_sell = getattr(cfg, 'k_sell', 0.3)
        mean_half = float(np.nanmean(self.spA)) / 2.0
        lam_th = A_buy  * np.exp(-k_buy  * mean_half) + \
                 A_sell * np.exp(-k_sell * mean_half)
        ax1.axhline(lam_th, color=_MID, lw=1.2, ls="--",
                    label=f"Theoretical λ = {lam_th:.4f}/s  (A·e^(−k·δ))")
        _ax_style(ax1, xlabel="Time (s)", ylabel="fills / s",
                  title="Rolling fill rate  (10-min window)  vs theoretical λ")
        ax1.legend(fontsize=7.5)

        # ── Fill size histogram ─────────────────────────────────────────────
        if len(mf):
            ax2.hist(mf["size"] / 1e3, bins=40, color=_INV,
                     edgecolor=_BG, alpha=0.85)
        ax2.set_xlabel("Fill size (kEUR)", color=_TXT)
        ax2.set_ylabel("Count", color=_TXT)
        _ax_style(ax2, title="Fill size distribution")

        # ── Rolling direction imbalance ─────────────────────────────────────
        if len(mf):
            mf2 = mf.copy()
            t_col = "t" if "t" in mf2.columns else "step"
            mf2 = mf2.sort_values(t_col)
            mf2["signed"] = np.where(mf2["direction"] == "buy", 1.0, -1.0)
            roll_imb = mf2["signed"].rolling(50, min_periods=1).mean()
            ax3.plot(mf2[t_col].values, roll_imb.values,
                     color=_SPR, lw=0.8, label="Rolling imbalance (buy=+1, sell=−1)")
            ax3.fill_between(mf2[t_col].values, 0, roll_imb.values,
                             where=roll_imb > 0, color=_BID, alpha=0.2)
            ax3.fill_between(mf2[t_col].values, 0, roll_imb.values,
                             where=roll_imb < 0, color=_ASK, alpha=0.2)
            ax3.axhline(0, color=_REF, lw=0.8)
            _vlines(ax3, self.hedge_t)
        _ax_style(ax3, xlabel="Time (s)", ylabel="Imbalance",
                  title="Order-flow imbalance  (50-fill rolling)  — should mean-revert around 0")
        if len(mf):
            ax3.legend(fontsize=7.5)

        # ── Level utilisation ───────────────────────────────────────────────
        if len(mf):
            lvl_counts = mf["level"].value_counts().sort_index()
            bars = ax4.bar(lvl_counts.index.astype(str), lvl_counts.values,
                           color=_MID, edgecolor=_BG, alpha=0.85)
            for bar, v in zip(bars, lvl_counts.values):
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5, str(v),
                         ha="center", va="bottom", fontsize=7, color=_TXT)
        _ax_style(ax4, xlabel="Level", ylabel="Fills",
                  title="Fills per quote level\n(L1 should dominate)")

        plt.tight_layout()
        plt.show()

    # ── Figure 4: Hedge Events ─────────────────────────────────────────────────

    def plot_hedges(self):
        """Deep dive on each hedge event."""
        sl     = self.sl
        hsteps = sorted(self.hedge_steps)
        dt     = self.ctrl.market_B.stock.time_step
        t_col  = "t" if "t" in sl.columns else "step"
        window = max(1, int(1800 / dt))   # ±30-min context window around hedge

        if len(hsteps) == 0:
            fig = _fig("Hedge Events — None triggered", figsize=(10, 3))
            ax  = fig.add_subplot(111)
            max_util = float((sl["inventory"].abs() * sl["fair_mid"]).max())
            ax.text(0.5, 0.5,
                    f"No hedges fired during this simulation.\n"
                    f"Max |inventory × mid| = ${max_util:,.0f}   "
                    f"(trigger = ${self.limit:,.0f} = {self.cfg.delta_limit:.0%} × K/2)\n"
                    f"Inventory stayed at {max_util/self.limit*100:.1f}% of the limit.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=_TXT,
                    bbox=dict(boxstyle="round", facecolor=_AX, edgecolor=_HED))
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        # Summary bar chart + context windows
        n_hedges = len(hsteps)
        ncols = min(n_hedges, 3)
        nrows = 1 + (n_hedges + ncols - 1) // ncols
        fig = _fig(f"Hedge Events  ({n_hedges} total)", figsize=(14, 3.5 * nrows))
        gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                                hspace=0.55, wspace=0.35)

        # ── Row 0: inventory at each hedge ─────────────────────────────────
        ax_sum = fig.add_subplot(gs[0, :])
        hedge_inv = []
        for hs in hsteps:
            row = sl[sl["step"] <= hs].iloc[-1]
            hedge_inv.append(abs(row["inventory"] * row["fair_mid"]))

        colors = [_ASK if v > self.limit else _SPR for v in hedge_inv]
        bars = ax_sum.bar(range(n_hedges),
                          [v / 1e3 for v in hedge_inv],
                          color=colors, edgecolor=_BG)
        ax_sum.axhline(self.limit / 1e3, color=_ASK, lw=1.5, ls="--",
                       label=f"Trigger ${self.limit/1e3:.0f}k")
        for i, bar in enumerate(bars):
            ax_sum.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"H{i+1}", ha="center", va="bottom",
                        fontsize=8, color=_TXT)
        _ax_style(ax_sum, xlabel="Hedge #",
                  ylabel="|inv × mid|  ($k)",
                  title="Inventory notional (USD) at each hedge trigger")
        ax_sum.legend(fontsize=8)

        # ── Context windows ─────────────────────────────────────────────────
        for i, hs in enumerate(hsteps):
            row_i = 1 + i // ncols
            col_i = i % ncols
            ax = fig.add_subplot(gs[row_i, col_i])

            lo = max(0, hs - window)
            hi = min(len(sl) - 1, hs + window)
            chunk = sl.iloc[lo:hi]
            tc    = chunk[t_col].values
            inv_c = chunk["inventory"].values

            ax.plot(tc, inv_c / 1e3, color=_INV, lw=0.9)
            ax.axhline( self.limit / 1e3, color=_ASK, lw=1, ls="--")
            ax.axhline(-self.limit / 1e3, color=_ASK, lw=1, ls="--")
            ax.axhline(0, color=_REF, lw=0.5)

            # mark the hedge step
            hs_t = float(sl[sl["step"] == hs][t_col].iloc[0]) if len(sl[sl["step"] == hs]) else hs
            ax.axvline(hs_t, color=_HED, lw=1.5, ls="--")

            idx = np.searchsorted(chunk["step"].values, hs, side="right") - 1
            if 0 <= idx < len(inv_c):
                inv_at = inv_c[idx]
                ax.text(0.97, 0.95, f"inv={inv_at/1e3:+.1f}k",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, color=_HED)

            _ax_style(ax, xlabel="Time (s)", ylabel="kEUR",
                      title=f"Hedge {i+1}  (±30 min)")

        plt.tight_layout()
        plt.show()

    # ── Figure 5: Parameter Scorecard ─────────────────────────────────────────

    def plot_param_scorecard(self):
        """Colour-coded parameter health check with tuning suggestions."""
        sl  = self.sl
        th  = self.th
        mf  = self.mm_fills
        cfg = self.cfg
        dt  = self.ctrl.market_B.stock.time_step

        # Compute metrics
        mean_spA = float(np.nanmean(self.spA))
        mean_spB = float(np.nanmean(self.spB))
        premium  = mean_spA - mean_spB

        inception_pnl = float(mf["cash_flow"].sum()) if len(mf) else 0.0
        total_fees    = float(th["fee_cost"].abs().sum()) if len(th) else 0.0
        edge_fee_ratio = inception_pnl / total_fees if total_fees > 0 else float("inf")

        A_buy  = getattr(cfg, 'A_buy',  0.007)
        A_sell = getattr(cfg, 'A_sell', 0.007)
        k_buy  = getattr(cfg, 'k_buy',  0.3)
        k_sell = getattr(cfg, 'k_sell', 0.3)
        lam_th = (A_buy  * np.exp(-k_buy  * mean_spA / 2) +
                  A_sell * np.exp(-k_sell * mean_spA / 2))
        obs_rate = len(mf) / (len(sl) * dt) if len(sl) > 0 else 0.0
        flow_ratio = obs_rate / lam_th if lam_th > 0 else 0.0

        inv_usd = (sl["inventory"].abs() * sl["fair_mid"]).values
        max_util = float(inv_usd.max()) / self.limit
        avg_util = float(inv_usd.mean()) / self.limit

        n_hedges_day = len(self.hedge_steps) / self.n_days

        fills_day = len(mf) / self.n_days

        # Scoring: (label, value_str, score 0-2, suggestion)
        # score: 0=red (problem), 1=yellow (watch), 2=green (ok)
        metrics = []

        # 1. Spread premium
        if premium < 1.0:
            sc, tip = 0, f"Premium too thin ({premium:.2f} bps). Consider raising gamma."
        elif premium > 8.0:
            sc, tip = 1, f"Premium wide ({premium:.2f} bps). May lose fills. Try reducing gamma."
        else:
            sc, tip = 2, "Spread premium looks healthy."
        metrics.append(("Spread premium\nA − B", f"{premium:.2f} bps", sc, tip))

        # 2. Fill rate vs theory
        if flow_ratio < 0.3:
            sc, tip = 0, f"Observed rate ({obs_rate:.4f}/s) << theoretical ({lam_th:.4f}/s). Spread may be too wide, or A/k wrong."
        elif flow_ratio > 2.0:
            sc, tip = 1, f"Observed >> theoretical. Client flow params (A, k) likely under-estimated."
        else:
            sc, tip = 2, f"Observed ≈ theoretical ({flow_ratio:.2f}× ratio)."
        metrics.append(("Fill rate\nobs / theoretical", f"{flow_ratio:.2f}×", sc, tip))

        # 3. Hedge frequency
        if n_hedges_day > 3:
            sc, tip = 0, f"{n_hedges_day:.1f} hedges/day is high. Raise delta_limit or tighten gamma."
        elif n_hedges_day < 0.1 and max_util < 0.3:
            sc, tip = 1, f"No hedges + max util {max_util:.0%}. delta_limit may be too loose."
        else:
            sc, tip = 2, f"{n_hedges_day:.1f} hedges/day, max util {max_util:.0%}."
        metrics.append(("Hedge frequency\n(per day)", f"{n_hedges_day:.2f}", sc, tip))

        # 4. Avg inventory utilisation (max is tautologically 100% when a hedge fires)
        if len(self.hedge_steps) > 0:
            # Hedge fired → max util is always ~100% by construction. Judge avg instead.
            if avg_util > 0.5:
                sc, tip = 1, f"Avg util {avg_util:.0%} — inventory spends a lot of time near the trigger. Consider raising delta_limit or reducing gamma."
            else:
                sc, tip = 2, f"Hedge fired as designed. Avg util {avg_util:.0%} — inventory is well-managed between hedges."
        else:
            # No hedge → max util tells you how close you got to firing
            if max_util > 0.85:
                sc, tip = 1, f"No hedge but max util {max_util:.0%} — inventory came close to the trigger. Watch this."
            elif max_util < 0.2:
                sc, tip = 1, f"Max util only {max_util:.0%} — trigger is very loose relative to actual inventory. Could tighten delta_limit."
            else:
                sc, tip = 2, f"No hedge needed. Max util {max_util:.0%} — good headroom."
        if len(self.hedge_steps) > 0:
            util_label = "Inventory utilisation\navg |inv| / trigger"
            util_val   = f"{avg_util:.0%}"
        else:
            util_label = "Inventory utilisation\nmax |inv| / trigger"
            util_val   = f"{max_util:.0%}"
        metrics.append((util_label, util_val, sc, tip))

        # 5. Edge / fee ratio
        if edge_fee_ratio < 1.5:
            sc, tip = 0, f"Inception spread / fees = {edge_fee_ratio:.1f}×. Fees eating edge. Widen spread or reduce hedging."
        elif edge_fee_ratio < 3.0:
            sc, tip = 1, f"Ratio {edge_fee_ratio:.1f}× is acceptable but tight."
        else:
            sc, tip = 2, f"Ratio {edge_fee_ratio:.1f}× — fees well covered by edge."
        metrics.append(("Edge / fees\n(inception spread / total fees)", f"{edge_fee_ratio:.1f}×", sc, tip))

        # 6. Fills per day
        if fills_day < 50:
            sc, tip = 0, f"Only {fills_day:.0f} fills/day. Very low liquidity capture. Raise A or lower gamma."
        elif fills_day > 2000:
            sc, tip = 1, f"{fills_day:.0f} fills/day is high. Watch inventory build-up."
        else:
            sc, tip = 2, f"{fills_day:.0f} fills/day — healthy."
        metrics.append(("Fills per day", f"{fills_day:.0f}", sc, tip))

        # ── Plot ────────────────────────────────────────────────────────────
        n = len(metrics)
        fig = _fig("Parameter Scorecard  —  Tuning Recommendations", figsize=(14, 6))
        gs  = gridspec.GridSpec(2, (n + 1) // 2, figure=fig,
                                hspace=0.6, wspace=0.4)

        _COLORS = {0: "#f85149", 1: "#ffa657", 2: "#3fb950"}
        _LABELS = {0: "PROBLEM", 1: "WATCH", 2: "OK"}

        for i, (label, val, score, tip) in enumerate(metrics):
            row = i // ((n + 1) // 2)
            col = i % ((n + 1) // 2)
            ax  = fig.add_subplot(gs[row, col])
            ax.set_facecolor(_AX)
            ax.axis("off")

            c = _COLORS[score]
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor=c + "22", edgecolor=c, lw=2,
                                       clip_on=False))
            ax.text(0.5, 0.82, label, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5, color=_TXT,
                    fontweight="bold")
            ax.text(0.5, 0.57, val, transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color=c,
                    fontweight="bold")
            ax.text(0.5, 0.38, _LABELS[score], transform=ax.transAxes,
                    ha="center", va="center", fontsize=8,
                    color=c, fontweight="bold")
            ax.text(0.5, 0.13, tip, transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color=_TXT,
                    wrap=True, style="italic")

        plt.tight_layout()
        plt.show()

    # ── Full report ────────────────────────────────────────────────────────────

    def report(self):
        self.plot_overview()
        self.plot_spread_and_skew()
        self.plot_fills()
        self.plot_hedges()
        self.plot_param_scorecard()
