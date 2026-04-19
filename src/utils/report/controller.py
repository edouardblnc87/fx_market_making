from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List

from ..market_simulator.market import Market
from ..order_book.order_book_impl import Order_book, Order
from ..market_maker.quoter import Quoter
from .pnl_tracker import PnLTracker
from tqdm import tqdm 

class Controller:
    """
    Simulation controller for the market-making backtest.

    Holds every simulation component in one place, runs the step loop,
    records a lightweight state snapshot at each step, and generates the
    backtesting report.

    Parameters
    ----------
    market_B: Market — reference market B (75% volume, 200 ms latency)
    market_C: Market — reference market C (25% volume, 170 ms latency)
    book: Order_book — market A order book
    quoter: Quoter — market-making logic
    client_flow_fn : callable(step, t, mid_A, best_bid_A, best_ask_A, dt)
                     → Iterable[Order]
                     Called once per step to generate client orders that are
                     immediately routed through the book.

    Usage
    -----
    ctrl = Controller(market_B, market_C, book, quoter, gen.generate_step)
    ctrl.simulate()
    ctrl.report()
    """

    def __init__(self, market_B: Market, market_C: Market, book: Order_book, quoter: Quoter, client_flow_fn: Callable) -> None:
        self.market_B = market_B
        self.market_C = market_C
        self.book = book
        self.quoter = quoter
        self.client_flow_fn = client_flow_fn
        self.number_order = []
        # Cache dt to avoid two chained attribute lookups per step.
        self._dt: float = market_B.stock.time_step

        # Pre-allocated numpy arrays (set by simulate(), None until then).
        self._log_arrays: dict | None = None
        self._log_ptr: int = 0

        # Fallback for manual step() calls outside simulate().
        self._step_log: List[dict] = []

        self._n_fills_prev: int = 0       # track fills-per-step delta
        self._n_quotes_posted: int = 0    # cumulative quotes sent to book


    # Simulation loop

    def step(self, step: int, t: float) -> None:
        """
        Advance the simulation by one step.

        Order of operations:
          1. Tick the order book (age resting orders).
          2. Compute MM quotes; cancel stale ones; post new ones.
          3. Derive market A best bid/ask from resting MM orders.
          4. Generate and route client orders (triggers matching + FillEvents).
          5. Execute hedge on B/C if delta limit is breached.
          6. Log the step state.
        """
        dt = getattr(self, '_dt', self.market_B.stock.time_step)

        self.book.tick(step)

        # Single call: mm_resting_orders returns the live dict reference.
        # compute_quotes reads it before we cancel/post; the same reference
        # then reflects the updated state for the bid/ask computation below.
        resting = self.book.mm_resting_orders
        quotes, cancels = self.quoter.compute_quotes(step, t, resting)
        self.book.cancel_orders(cancels)
        self.book.post_mm_quotes(quotes)
        self._n_quotes_posted += len(quotes)

        # Market A best bid/ask from resting MM orders (single pass).
        bids = []
        asks = []
        for v in resting.values():
            if v['direction'] == 'buy':
                bids.append(v['price'])
            else:
                asks.append(v['price'])

        if bids and asks:
            best_bid_A = max(bids)
            best_ask_A = min(asks)
            mid_A = (best_bid_A + best_ask_A) / 2.0
            a = self.client_flow_fn(step, t, mid_A, best_bid_A, best_ask_A, dt)

            self.number_order.append(a)
            for order in a:
                
                self.book.route_client_order(order)
            self.quoter.execute_hedge(step, t, mid_A)
        else:
            best_bid_A = np.nan
            best_ask_A = np.nan

        self._log_step(step, t, best_bid_A, best_ask_A)

    def simulate(self, limit: int | None = None) -> None:
        """Run the full simulation from step 0 to n_steps − 1."""
        n  = limit if limit is not None else self.market_B.stock.n_steps
        dt = getattr(self, '_dt', self.market_B.stock.time_step)

        # Pre-allocate contiguous numpy arrays for the step log.
        # Writing by index is ~10× faster than appending dicts to a list.
        self._log_arrays = {
            'step':                np.empty(n, dtype=np.int64),
            't':                   np.empty(n),
            'bid_A':               np.empty(n),
            'ask_A':               np.empty(n),
            'mid_A':               np.empty(n),
            'bid_B':               np.empty(n),
            'ask_B':               np.empty(n),
            'mid_B':               np.empty(n),
            'bid_C':               np.empty(n),
            'ask_C':               np.empty(n),
            'mid_C':               np.empty(n),
            'fair_mid':            np.empty(n),
            'inventory':           np.empty(n),
            'n_mm_resting':        np.empty(n, dtype=np.int64),
            'fills_this_step':     np.empty(n, dtype=np.int64),
            'total_quotes_posted': np.empty(n, dtype=np.int64),
        }
        self._log_ptr = 0
        self._n_fills_prev    = 0
        self._n_quotes_posted = 0
        self._step_log        = []
        # Reset quoter fill history so re-running doesn't accumulate stale entries.
        self.quoter._fill_history.clear()
        self.quoter._pending_fills.clear()
        self.quoter._pending_topups.clear()

        if limit is not None:
            for s in tqdm(range(n)):
                self.step(s, s * dt)
        else:
            for s in range(n):
                self.step(s, s * dt)

    def _log_step(self, step: int, t: float, bid_A: float, ask_A: float) -> None:
        """
        Write a lightweight state snapshot for this step.

        Uses pre-allocated numpy arrays when available (fast path set by
        simulate()); falls back to list-of-dicts for manual step() calls.

        Columns logged
        --------------
        t, step
        bid_A, ask_A, mid_A       — market A best resting MM quotes
        bid_B, ask_B, mid_B       — market B prices at this step
        bid_C, ask_C, mid_C       — market C prices at this step
        fair_mid                  — weighted B/C mid (quoter reference)
        inventory                 — current EUR inventory
        n_mm_resting              — number of resting MM orders on A
        fills_this_step           — new fills (MM + hedge) since last step
        total_quotes_posted       — cumulative MM quotes sent to book
        """
        bid_B = float(self.market_B.bid_price[step])
        ask_B = float(self.market_B.ask_price[step])
        bid_C = float(self.market_C.bid_price[step])
        ask_C = float(self.market_C.ask_price[step])
        mid_B = (bid_B + ask_B) * 0.5
        mid_C = (bid_C + ask_C) * 0.5

        wb = self.quoter.cfg.weight_B
        wc = self.quoter.cfg.weight_C
        fair_mid = wb * mid_B + wc * mid_C

        n_fills_now = len(self.quoter._fill_history)
        fills_this_step = n_fills_now - self._n_fills_prev
        self._n_fills_prev = n_fills_now

        mid_A = (bid_A + ask_A) * 0.5 if not np.isnan(bid_A) else np.nan

        if getattr(self, '_log_arrays', None) is not None:
            a = self._log_arrays
            a['step'][step]                = step
            a['t'][step]                   = t
            a['bid_A'][step]               = bid_A
            a['ask_A'][step]               = ask_A
            a['mid_A'][step]               = mid_A
            a['bid_B'][step]               = bid_B
            a['ask_B'][step]               = ask_B
            a['mid_B'][step]               = mid_B
            a['bid_C'][step]               = bid_C
            a['ask_C'][step]               = ask_C
            a['mid_C'][step]               = mid_C
            a['fair_mid'][step]            = fair_mid
            a['inventory'][step]           = self.quoter.inventory
            a['n_mm_resting'][step]        = len(self.book._mm_resting)
            a['fills_this_step'][step]     = fills_this_step
            a['total_quotes_posted'][step] = self._n_quotes_posted
            self._log_ptr = step + 1
        else:
            self._step_log.append({
                'step': step,
                't': t,
                'bid_A': bid_A,
                'ask_A': ask_A,
                'mid_A': mid_A,
                'bid_B': bid_B,
                'ask_B': ask_B,
                'mid_B': mid_B,
                'bid_C': bid_C,
                'ask_C': ask_C,
                'mid_C': mid_C,
                'fair_mid': fair_mid,
                'inventory': self.quoter.inventory,
                'n_mm_resting': len(self.book._mm_resting),
                'fills_this_step': fills_this_step,
                'total_quotes_posted': self._n_quotes_posted,
            })

    # Properties and report generation

    @property
    def step_log(self) -> pd.DataFrame:
        """All logged steps as a DataFrame (one row per simulation step)."""
        if getattr(self, '_log_arrays', None) is not None:
            n = self._log_ptr
            return pd.DataFrame({k: v[:n] for k, v in self._log_arrays.items()})
        return pd.DataFrame(self._step_log)

    @property
    def trade_history(self) -> pd.DataFrame:
        """All fills (MM fills + hedge legs) — delegates to Quoter."""
        return self.quoter.trade_history

    def _current_fair_mid(self) -> float:
        if getattr(self, '_log_arrays', None) is not None and self._log_ptr > 0:
            return float(self._log_arrays['fair_mid'][self._log_ptr - 1])
        if self._step_log:
            return self._step_log[-1]['fair_mid']
        df = self.trade_history
        return float(df['fair_mid'].iloc[-1]) if not df.empty else 1.0

    # P&L (delegates to PnLTracker)

    def pnl_report(self) -> dict:
        """
        Full P&L breakdown at the current simulation state.
        Delegates entirely to PnLTracker.report().
        """
        df = self.trade_history
        if df.empty:
            return {}
        return PnLTracker.report(df, self._current_fair_mid())

    # Backtesting report

    @staticmethod
    def _ds(log: pd.DataFrame, max_pts: int = 2000) -> pd.DataFrame:
        """Downsample log to at most max_pts rows for plotting."""
        stride = max(1, len(log) // max_pts)
        return log.iloc[::stride]

    def plot_market_quotes(self) -> None:
        """
        Plot prices and spread-in-bps for all three markets.

        Four stacked panels (A, B, C, Spread bps) sharing the time axis.
        Market A shows the best resting MM quotes; B and C show the simulated
        reference prices. The bottom panel overlays the bid-ask spread in bps
        for all three markets — useful to spot when quotes are too tight/wide.
        """
        log = self.step_log
        if log.empty:
            print("No log data — run simulate() first.")
            return

        s = self._ds(log)
        t = s['t']

        s = s.copy()
        s['spd_A'] = (s['ask_A'] - s['bid_A']) / s['mid_A'].replace(0, np.nan) * 1e4
        s['spd_B'] = (s['ask_B'] - s['bid_B']) / s['mid_B'].replace(0, np.nan) * 1e4
        s['spd_C'] = (s['ask_C'] - s['bid_C']) / s['mid_C'].replace(0, np.nan) * 1e4

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        fig.patch.set_facecolor('#111111')

        def _style(ax):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#444444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444444')
            ax.spines['bottom'].set_color('#444444')

        # ── Panel A: absolute price — mid, bid, ask ──
        ax_A = axes[0]
        _style(ax_A)
        ax_A.plot(t, s['mid_A'], color='#ffffff', linewidth=0.8, label='Mid', zorder=3)
        ax_A.plot(t, s['bid_A'], color='#00ff88', linewidth=0.5, linestyle='--', label='Bid', zorder=2)
        ax_A.plot(t, s['ask_A'], color='#ff4444', linewidth=0.5, linestyle='--', label='Ask', zorder=2)
        ax_A.fill_between(t, s['bid_A'], s['ask_A'], alpha=0.07, color='#ffffff')
        ax_A.set_title('Market A — MM quoted (this exchange)', color='white', fontsize=12)
        ax_A.set_ylabel('Price', color='white')
        ax_A.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white',
                    fontsize=8, loc='upper left')

        # ── Panels B and C: standard absolute price ──
        ref_panels = [
            (axes[1], 'B — reference (75% vol, 200 ms)',
             'bid_B', 'ask_B', 'mid_B', '#44bbff', '#ff8844', '#ffcc00'),
            (axes[2], 'C — reference (25% vol, 170 ms)',
             'bid_C', 'ask_C', 'mid_C', '#aa44ff', '#ff44aa', '#aaffaa'),
        ]
        for ax, title, bc, ac, mc, cbid, cask, cmid in ref_panels:
            _style(ax)
            ax.plot(t, s[mc], color=cmid, linewidth=0.8, label='Mid', zorder=3)
            ax.plot(t, s[bc], color=cbid, linewidth=0.5, linestyle='--', label='Bid', zorder=2)
            ax.plot(t, s[ac], color=cask, linewidth=0.5, linestyle='--', label='Ask', zorder=2)
            ax.fill_between(t, s[bc], s[ac], alpha=0.07, color=cmid)
            ax.set_title(f'Market {title}', color='white', fontsize=12)
            ax.set_ylabel('Price', color='white')
            ax.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white',
                      fontsize=8, loc='upper left')

        # Panel 4 — spread in bps
        ax4 = axes[3]
        ax4.set_facecolor('#111111')
        ax4.tick_params(colors='white')
        ax4.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#444444')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color('#444444')
        ax4.spines['bottom'].set_color('#444444')
        ax4.plot(t, s['spd_A'], color='#ffffff', linewidth=0.7, label='A (MM)', zorder=3)
        ax4.plot(t, s['spd_B'], color='#ffcc00', linewidth=0.7, alpha=0.8, label='B', zorder=2)
        ax4.plot(t, s['spd_C'], color='#aa44ff', linewidth=0.7, alpha=0.8, label='C', zorder=2)
        ax4.set_title('Bid-Ask Spread — all markets (bps)', color='white', fontsize=12)
        ax4.set_ylabel('Spread (bps)', color='white')
        ax4.set_xlabel('Time (s)', color='white')
        ax4.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=8,
                   loc='upper right')

        n_days = round(log['t'].iloc[-1] / 86400, 1) if len(log) > 1 else '?'
        plt.suptitle(f'Bid / Ask / Mid — All Markets  ({n_days} days)',
                     color='white', fontsize=14, y=1.005)
        plt.tight_layout()
        plt.show()

    def plot_top_trades(self, n: int = 10) -> None:
        """
        Plot the bid/ask/mid evolution for market A with the top n fills by size marked.

        Buy fills shown as up-arrows (green), sell fills as down-arrows (red).
        Marker area scales with fill size.
        """
        log = self.step_log
        df = self.trade_history
        if log.empty or df.empty:
            print("No data — run simulate() first.")
            return

        mm = df[~df['is_hedge']]
        top = mm.nlargest(min(n, len(mm)), 'size')

        s = self._ds(log)
        t = s['t']
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#111111')
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#444444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')

        ax.plot(t, s['mid_A'], color='#ffffff', linewidth=0.8, label='Mid A', zorder=2)
        ax.plot(t, s['bid_A'], color='#00ff88', linewidth=0.5, linestyle='--',
                label='Bid A', zorder=2)
        ax.plot(t, s['ask_A'], color='#ff4444', linewidth=0.5, linestyle='--',
                label='Ask A', zorder=2)

        buys  = top[top['direction'] == 'buy']
        sells = top[top['direction'] == 'sell']
        max_s = top['size'].max() if len(top) else 1.0

        def _ms(sz): return 40 + 180 * (sz / max_s)

        if len(buys):
            ax.scatter(buys['t'], buys['price'], marker='^', color='#00ff88',
                       s=_ms(buys['size']), zorder=5, label=f'Buy fill (top {n})')
        if len(sells):
            ax.scatter(sells['t'], sells['price'], marker='v', color='#ff4444',
                       s=_ms(sells['size']), zorder=5, label=f'Sell fill (top {n})')

        ax.set_title(f'Market A — Bid/Ask/Mid with top {n} fills by size',
                     color='white', fontsize=13)
        ax.set_ylabel('Price', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_price_inventory(self) -> None:
        """
        Task 7 — EUR/USD fair_mid price evolution (left axis) alongside
        EUR inventory (right axis) with ±delta_limit lines.
        """
        log = self.step_log
        if log.empty:
            print("No log data — run simulate() first.")
            return

        s = self._ds(log)
        t = s['t']
        fig, ax1 = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#111111')
        ax1.set_facecolor('#111111')
        ax1.tick_params(colors='white')
        ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.4, color='#444444')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#ffcc00')
        ax1.spines['bottom'].set_color('#444444')

        ax1.plot(t, s['fair_mid'], color='#ffcc00', linewidth=0.8, label='EUR/USD fair mid')
        ax1.set_ylabel('EUR/USD price', color='#ffcc00')
        ax1.tick_params(axis='y', colors='#ffcc00')

        ax2 = ax1.twinx()
        ax2.set_facecolor('#111111')
        ax2.spines['right'].set_color('#ff9500')
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(colors='#ff9500')

        ax2.plot(t, s['inventory'], color='#ff9500', linewidth=0.8, label='Inventory (EUR)')
        ax2.axhline(0, color='#555', linewidth=0.6)

        limit = self.quoter.cfg.delta_limit * self.quoter.capital_K
        ax2.axhline( limit, color='#ff4444', linewidth=0.7, linestyle='--',
                    label=f'+{self.quoter.cfg.delta_limit:.0%} limit')
        ax2.axhline(-limit, color='#ff4444', linewidth=0.7, linestyle='--',
                    label=f'-{self.quoter.cfg.delta_limit:.0%} limit')
        ax2.set_ylabel('Inventory (EUR)', color='#ff9500')
        ax2.tick_params(axis='y', colors='#ff9500')

        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2,
                   facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)

        ax1.set_title('EUR/USD Price vs Inventory over Time', color='white', fontsize=13)
        ax1.set_xlabel('Time (s)', color='white')
        plt.tight_layout()
        plt.show()

    def plot_mtm_percentiles(self) -> None:
        """
        Plot the average, median, 5th and 95th percentile MtM P&L per trade
        as a function of time elapsed since trade inception.

        Uses PnLTracker.per_trade_mtm_evolution() to decompose aggregate MtM
        into per-fill contributions, then bins by relative time τ to build the
        percentile distribution across all MM fills.
        """
        df = self.trade_history
        if df.empty:
            print("No trade data — run simulate() first.")
            return

        mm_mask = ~df['is_hedge']
        if not mm_mask.any():
            print("No MM fills to analyse.")
            return

        t_vals = df['t'].values

        # Integer positions of MM fills (column indices in ev)
        mm_pos = [i for i in range(len(df)) if mm_mask.iloc[i]]

        # Cap at 500 fills to avoid building a huge matrix (n×n float64)
        max_sample = 500
        rng_sample = np.random.default_rng(0)
        if len(mm_pos) > max_sample:
            mm_pos = sorted(rng_sample.choice(mm_pos, size=max_sample, replace=False).tolist())

        # Build the evolution only for the sampled fill indices
        ev = PnLTracker.per_trade_mtm_evolution(df, fill_indices=mm_pos)

        records = []
        for col_idx, pos in enumerate(mm_pos):
            t_i = t_vals[pos]
            col = ev.iloc[:, col_idx].dropna()
            taus = col.index.to_numpy() - t_i
            for tau, val in zip(taus, col.values):
                records.append({'tau': tau, 'mtm': val})

        if not records:
            return

        rel = pd.DataFrame(records)
        max_tau = rel['tau'].max()
        n_bins = min(200, len(rel) // 10 + 1)
        bins = np.linspace(0.0, max_tau, n_bins + 1)
        rel['tau_bin'] = pd.cut(rel['tau'], bins=bins, labels=False)

        grp = rel.groupby('tau_bin')['mtm']
        tau_centers = (bins[:-1] + bins[1:]) / 2.0
        p05 = grp.quantile(0.05).reindex(range(n_bins)).values
        p50 = grp.quantile(0.50).reindex(range(n_bins)).values
        p95 = grp.quantile(0.95).reindex(range(n_bins)).values
        mean = grp.mean().reindex(range(n_bins)).values

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#111111')
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#444444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')

        ax.fill_between(tau_centers, p05, p95, alpha=0.20, color='#4499ff', label='5th – 95th percentile')
        ax.plot(tau_centers, p50, color='#4499ff', linewidth=1.1, label='Median')
        ax.plot(tau_centers, mean, color='#ffcc00', linewidth=1.1, linestyle='--', label='Mean')
        ax.axhline(0, color='#444', linewidth=0.6)

        n_sampled = len(mm_pos)
        ax.set_title(f'Per-trade MtM P&L since inception — {n_sampled} MM fills sample (USD)',
                     color='white', fontsize=13)
        ax.set_xlabel('τ  =  time since fill inception (s)', color='white')
        ax.set_ylabel('MtM contribution (USD)', color='white')
        ax.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)
        plt.tight_layout()
        plt.show()

    def fill_rate_analysis(self, plot: bool = True) -> dict:
        """
        Fill-rate analysis for MM quotes on exchange A.

        Returns
        -------
        dict with:
          total_mm_fills      : number of MM fill events (partial + full)
          total_quotes_posted : cumulative quotes submitted to the book
          overall_fill_rate   : total_mm_fills / total_quotes_posted
          fills_by_level      : Series — fill count per ladder level (1 = best)
          fill_rate_by_level  : Series — fraction of fills at each level vs level 1
          avg_fill_size_eur   : mean fill size in EUR
          full_fill_pct       : fraction of MM fills that were full fills
          partial_fill_pct    : fraction that were partial fills

        If plot=True, draws a two-panel bar chart (fill count + relative rate by level).
        """
        df = self.trade_history
        mm = df[~df['is_hedge']] if not df.empty else pd.DataFrame()

        total_quotes = self._n_quotes_posted
        total_fills  = len(mm)
        overall_rate = total_fills / total_quotes if total_quotes > 0 else 0.0

        fills_by_level = mm['level'].value_counts().sort_index() if not mm.empty else pd.Series(dtype=float)
        fill_rate_by_level = (fills_by_level / fills_by_level.iloc[0]
                              if not fills_by_level.empty else pd.Series(dtype=float))

        stats = {
            'total_mm_fills': total_fills,
            'total_quotes_posted': total_quotes,
            'overall_fill_rate': overall_rate,
            'fills_by_level': fills_by_level,
            'fill_rate_by_level': fill_rate_by_level,
            'avg_fill_size_eur': float(mm['size'].mean()) if not mm.empty else 0.0,
            'full_fill_pct': float(mm['is_full_fill'].mean()) if not mm.empty else 0.0,
            'partial_fill_pct': float((~mm['is_full_fill']).mean()) if not mm.empty else 0.0,
        }

        if plot and not fills_by_level.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#111111')
            for ax in axes:
                ax.set_facecolor('#111111')
                ax.tick_params(colors='white')
                ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#444444')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#444444')
                ax.spines['bottom'].set_color('#444444')

            axes[0].bar(fills_by_level.index, fills_by_level.values, color='#4499ff', alpha=0.85, edgecolor='#111111')
            axes[0].set_title('Fill count by ladder level', color='white', fontsize=12)
            axes[0].set_xlabel('Level (1 = best quote)', color='white')
            axes[0].set_ylabel('Number of fills', color='white')

            axes[1].bar(fill_rate_by_level.index, fill_rate_by_level.values, color='#00ff88', alpha=0.85, edgecolor='#111111')
            axes[1].set_title('Relative fill rate by level (normalised to level 1)', color='white', fontsize=12)
            axes[1].set_xlabel('Level (1 = best quote)', color='white')
            axes[1].set_ylabel('Relative rate', color='white')

            plt.suptitle(
                f'Fill-rate analysis | overall = {overall_rate:.4%} | '
                f'full fills = {stats["full_fill_pct"]:.1%}',
                color='white', fontsize=12,
            )
            plt.tight_layout()
            plt.show()

        return stats

    def report(self) -> None:
        """
        Generate the full backtesting report.

        Prints the P&L summary, then renders all five plots:
          1. Market quotes for A, B, C
          2. Top 10 trades on market A
          3. EUR/USD price vs inventory
          4. 4-panel session P&L chart (from PnLTracker)
          5. Per-trade MtM percentiles
          6. Fill-rate analysis
        """
        df = self.trade_history
        if df.empty:
            print("No trade data — run simulate() first.")
            return

        current_mid = self._current_fair_mid()
        rep = PnLTracker.report(df, current_mid)

        log = self.step_log
        n_days = round(log['t'].iloc[-1] / 86400, 1) if len(log) > 1 else '?'
        n_steps = len(log)
        dt = self.market_B.stock.time_step
        fills_per_day = rep['n_mm_fills'] / max(float(n_days), 1e-9)

        # Spread diagnostics from step log
        spd_A = (log['ask_A'] - log['bid_A']) / log['mid_A'] * 1e4
        spd_B = (log['ask_B'] - log['bid_B']) / log['mid_B'] * 1e4
        spd_C = (log['ask_C'] - log['mid_C']) * 2 / log['mid_C'] * 1e4  # approx

        width = 68
        print("═" * width)
        print(f"  BACKTESTING REPORT — Phase 1   ({n_days} days, dt={dt:.1f}s, {n_steps:,} steps)")
        print("═" * width)
        print(f"  {'Total MtM P&L':<34}  {rep['total_mtm_pnl']:>+14.2f}  USD")
        print(f"  {'  Realized cash P&L':<34}  {rep['realized_pnl']:>+14.2f}  USD")
        print(f"  {'  Unrealized (open inventory × mid)':<34}  {rep['unrealized_pnl']:>+14.2f}  USD")
        print("─" * width)
        print(f"  {'Inception spread P&L':<34}  {rep['inception_spread_pnl']:>+14.2f}  USD")
        print(f"  {'  (spread captured at fill time)':<34}")
        print(f"  {'Inventory revaluation P&L':<34}  {rep['inventory_revaluation_pnl']:>+14.2f}  USD")
        print(f"  {'  (mid drift on open EUR position)':<34}")
        print(f"  {'Total fees paid':<34}  {rep['total_fees']:>14.2f}  USD")
        print(f"  {'  Maker fees (exchange A)':<34}  {rep['mm_maker_fees']:>14.2f}  USD")
        print(f"  {'  Taker fees (hedge B/C)':<34}  {rep['hedge_taker_fees']:>14.2f}  USD")
        print("─" * width)
        print(f"  {'MM fills':<34}  {rep['n_mm_fills']:>14,}  ({fills_per_day:,.0f}/day)")
        print(f"  {'Hedge legs':<34}  {rep['n_hedges']:>14,}")
        print(f"  {'Final inventory':<34}  {rep['final_inventory_eur']:>14,.0f}  EUR")
        print("─" * width)
        print(f"  {'Spread A (MM quoted, bps)':<34}  mean={spd_A.mean():.2f}  "
              f"σ={spd_A.std():.2f}  [{spd_A.min():.2f}, {spd_A.max():.2f}]")
        print(f"  {'Spread B (reference, bps)':<34}  mean={spd_B.mean():.2f}  "
              f"σ={spd_B.std():.2f}")
        print("═" * width)

        self.plot_market_quotes()
        self.plot_top_trades(n=10)
        self.plot_price_inventory()
        PnLTracker.plot(df, current_mid, capital_K=self.quoter.capital_K,
                        delta_limit=self.quoter.cfg.delta_limit, step_log=self.step_log)
        self.plot_mtm_percentiles()
        self.fill_rate_analysis(plot=True)
