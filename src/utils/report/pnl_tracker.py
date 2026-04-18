from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PnLTracker:
    """
    Static computation class for P&L analysis of a market-making session.

    All methods operate on the DataFrame returned by Quoter.trade_history.
    No state — purely functions of the trade log.

    P&L decomposition identities
    -----------------------------
    Total MtM P&L  =  realized_pnl + unrealized_pnl
                   =  inception_spread_pnl + inventory_revaluation_pnl - total_fees

    - realized_pnl         : cash already locked in (sum of cash_flow column)
    - unrealized_pnl        : open EUR inventory valued at current mid in USD
    - inception_spread_pnl  : spread captured at the moment each MM fill happened
    - inventory_revaluation : P&L from fair_mid moving after fills (residual)
    - total_fees            : all fees paid (maker on A + taker on B/C)
    """

    @staticmethod
    def augment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns to the trade history DataFrame.

        Columns added
        -------------
        cum_cash  : cumulative realized cash P&L (USD)
        cum_fees  : cumulative fees paid (USD)
        mtm_pnl   : running MtM P&L per row = cum_cash + inventory_after * fair_mid
                    (uses the fair_mid at the time of each fill as the mark price)
        """
        df = df.copy()
        df['cum_cash'] = df['cash_flow'].cumsum()
        df['cum_fees'] = df['fee_cost'].cumsum()
        df['mtm_pnl'] = df['cum_cash'] + df['inventory_after'] * df['fair_mid']
        return df

    @staticmethod
    def realized_pnl(df: pd.DataFrame) -> float:
        """Total realized USD cash P&L = sum of all cash_flow entries."""
        return float(df['cash_flow'].sum())

    @staticmethod
    def unrealized_pnl(df: pd.DataFrame, current_mid: float) -> float:
        """
        USD value of the open EUR inventory at current_mid.
        Positive when long EUR and price is up, negative when short.
        """
        if df.empty:
            return 0.0
        inventory = float(df['inventory_after'].iloc[-1])
        return inventory * current_mid

    @staticmethod
    def mtm_pnl(df: pd.DataFrame, current_mid: float) -> float:
        """
        Total mark-to-market P&L = realized + unrealized.
        This is the true economic P&L if the position were closed now at current_mid.
        """
        return PnLTracker.realized_pnl(df) + PnLTracker.unrealized_pnl(df, current_mid)

    @staticmethod
    def inception_spread_pnl(df: pd.DataFrame) -> float:
        """
        Spread P&L captured at the moment of each MM fill on exchange A.

        For a sell fill: we sold EUR above fair mid → profit = (price - fair_mid) * size
        For a buy fill:  we bought EUR below fair mid → profit = (fair_mid - price) * size

        Hedge legs are excluded — their economics are captured in cash_flow.
        """
        mm = df[~df['is_hedge']]
        if mm.empty:
            return 0.0
        sells = mm[mm['direction'] == 'sell']
        buys = mm[mm['direction'] == 'buy']
        spread_sells = ((sells['price'] - sells['fair_mid']) * sells['size']).sum()
        spread_buys = ((buys['fair_mid'] - buys['price']) * buys['size']).sum()
        return float(spread_sells + spread_buys)

    @staticmethod
    def inventory_revaluation_pnl(df: pd.DataFrame, current_mid: float) -> float:
        """
        P&L from fair_mid moving after fills (mid-to-mid revaluation effect).
        This is the residual once inception spread and fees are stripped out:

            revaluation = total_MtM - inception_spread - total_fees

        Positive when inventory and price move in the same direction.
        """
        total = PnLTracker.mtm_pnl(df, current_mid)
        inception = PnLTracker.inception_spread_pnl(df)
        fees = float(df['fee_cost'].sum())
        return total - inception + fees

    @staticmethod
    def final_inventory_value(df: pd.DataFrame, current_mid: float) -> float:
        """USD value of the final EUR inventory at current_mid."""
        if df.empty:
            return 0.0
        return float(df['inventory_after'].iloc[-1]) * current_mid

    @staticmethod
    def per_trade_mtm_evolution(
        df: pd.DataFrame,
        fill_indices: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Decompose the aggregate MtM into per-fill contributions at each fill time.

        When fill_indices is None (default), all n fills are computed → shape (n, n).
        When fill_indices is given, only those column fills are evaluated → shape (n, k)
        where k = len(fill_indices).  Rows are still all n fill times.

        value[j, i] = cash_flow_i + Δinv_i × fair_mid_j   for j >= row_i, else NaN

        Identity (full matrix only): ev.sum(axis=1) == augment(df)['mtm_pnl']
        """
        if df.empty:
            return pd.DataFrame()
        inv_delta = df['inventory_after'].diff().fillna(df['inventory_after'].iloc[0]).values
        cash = df['cash_flow'].values
        mids = df['fair_mid'].values
        n = len(df)
        cols = list(range(n)) if fill_indices is None else fill_indices
        mat = np.full((n, len(cols)), np.nan)
        for col_idx, i in enumerate(cols):
            mat[i:, col_idx] = cash[i] + inv_delta[i] * mids[i:]
        return pd.DataFrame(mat, index=df['t'].values, columns=cols)

    @staticmethod
    def report(df: pd.DataFrame, current_mid: float) -> dict:
        """
        Full P&L breakdown as a dict.

        Keys
        ----
        realized_pnl              : total realized cash P&L (USD)
        unrealized_pnl            : open inventory value at current_mid (USD)
        total_mtm_pnl             : realized + unrealized
        inception_spread_pnl      : spread captured at fill time
        inventory_revaluation_pnl : mid-to-mid revaluation effect
        total_fees                : all fees (maker A + taker B/C)
        mm_maker_fees             : fees on exchange A fills only
        hedge_taker_fees          : fees on hedge legs (B and C)
        final_inventory_eur       : remaining EUR position
        final_inventory_usd       : remaining EUR position converted to USD
        n_mm_fills                : number of MM fills on exchange A
        n_hedges                  : number of hedge legs executed
        n_total_trades            : total rows in trade history
        """
        if df.empty:
            return {k: 0.0 for k in [
                'realized_pnl', 'unrealized_pnl', 'total_mtm_pnl',
                'inception_spread_pnl', 'inventory_revaluation_pnl',
                'total_fees', 'mm_maker_fees', 'hedge_taker_fees',
                'final_inventory_eur', 'final_inventory_usd',
                'n_mm_fills', 'n_hedges', 'n_total_trades',
            ]}

        mm_fills = df[~df['is_hedge']]
        hedges = df[df['is_hedge']]

        realized = PnLTracker.realized_pnl(df)
        unrealized = PnLTracker.unrealized_pnl(df, current_mid)
        total_mtm = realized + unrealized
        inception = PnLTracker.inception_spread_pnl(df)
        total_fees = float(df['fee_cost'].sum())
        revaluation = total_mtm - inception + total_fees

        return {
            'realized_pnl':              realized,
            'unrealized_pnl':            unrealized,
            'total_mtm_pnl':             total_mtm,
            'inception_spread_pnl':      inception,
            'inventory_revaluation_pnl': revaluation,
            'total_fees':                total_fees,
            'mm_maker_fees':             float(mm_fills['fee_cost'].sum()),
            'hedge_taker_fees':          float(hedges['fee_cost'].sum()),
            'final_inventory_eur':       float(df['inventory_after'].iloc[-1]),
            'final_inventory_usd':       PnLTracker.final_inventory_value(df, current_mid),
            'n_mm_fills':                len(mm_fills),
            'n_hedges':                  len(hedges),
            'n_total_trades':            len(df),
        }

    @staticmethod
    def plot(df: pd.DataFrame, current_mid: float,
             capital_K: float | None = None, delta_limit: float = 0.90) -> None:
        """
        4-panel P&L and inventory chart.

        Panel 1 — MtM P&L vs realized cash P&L over time
        Panel 2 — Inventory (EUR) with optional ±delta_limit lines
        Panel 3 — P&L decomposition: running inception spread vs inventory revaluation
        Panel 4 — Cumulative fees split: maker (A) vs taker hedge (B/C)

        Parameters
        ----------
        df           : Quoter.trade_history DataFrame
        current_mid  : current fair mid price (used for unrealized P&L)
        capital_K    : total capital — if provided, draws ±delta_limit lines on inventory panel
        delta_limit  : fraction of capital_K for the inventory limit lines (default 0.90)
        """
        if df.empty:
            print("No trades to plot.")
            return

        aug = PnLTracker.augment(df)

        # Running inception spread per fill (MM fills on A only, hedges contribute 0)
        inception_per_fill = pd.Series(0.0, index=aug.index)
        mm_idx = aug[~aug['is_hedge']].index
        sells_idx = aug[(~aug['is_hedge']) & (aug['direction'] == 'sell')].index
        buys_idx  = aug[(~aug['is_hedge']) & (aug['direction'] == 'buy')].index
        inception_per_fill.loc[sells_idx] = (
            (aug.loc[sells_idx, 'price'] - aug.loc[sells_idx, 'fair_mid'])
            * aug.loc[sells_idx, 'size']
        )
        inception_per_fill.loc[buys_idx] = (
            (aug.loc[buys_idx, 'fair_mid'] - aug.loc[buys_idx, 'price'])
            * aug.loc[buys_idx, 'size']
        )
        aug['cum_inception'] = inception_per_fill.cumsum()
        aug['cum_revaluation'] = aug['mtm_pnl'] - aug['cum_inception'] + aug['cum_fees']

        # Cumulative fees split by venue
        maker_fees = df[~df['is_hedge']]['fee_cost'].cumsum().reindex(aug.index).ffill().fillna(0)
        hedge_fees = df[df['is_hedge']]['fee_cost'].cumsum().reindex(aug.index).ffill().fillna(0)

        hedges = aug[aug['is_hedge']]

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        fig.patch.set_facecolor('#111111')
        for ax in axes:
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#444444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444444')
            ax.spines['bottom'].set_color('#444444')

        # Panel 1 — MtM P&L vs realized cash
        axes[0].plot(aug['t'], aug['mtm_pnl'], color='#00ff88', linewidth=0.8, label='MtM P&L')
        axes[0].plot(aug['t'], aug['cum_cash'], color='#4499ff', linewidth=0.8,
                     linestyle='--', label='Realized cash')
        axes[0].axhline(0, color='#444', linewidth=0.6)
        if len(hedges):
            axes[0].scatter(hedges['t'], aug.loc[hedges.index, 'mtm_pnl'],
                            color='#ff4444', s=14, zorder=5, label='Hedge')
        axes[0].set_title('MtM P&L vs Realized Cash P&L (USD)', color='white', fontsize=13)
        axes[0].set_ylabel('P&L (USD)', color='white')
        axes[0].legend(facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)

        # Panel 2 — Inventory
        axes[1].plot(aug['t'], aug['inventory_after'], color='#ff9500', linewidth=0.8)
        axes[1].axhline(0, color='#444', linewidth=0.6)
        if capital_K is not None:
            lim = capital_K * delta_limit
            axes[1].axhline(lim,  color='#ff4444', linewidth=0.6, linestyle='--',
                            label=f'+{delta_limit:.0%} limit')
            axes[1].axhline(-lim, color='#ff4444', linewidth=0.6, linestyle='--',
                            label=f'-{delta_limit:.0%} limit')
            axes[1].legend(facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)
        axes[1].set_title('Inventory (EUR)', color='white', fontsize=13)
        axes[1].set_ylabel('EUR', color='white')

        # Panel 3 — P&L decomposition
        axes[2].plot(aug['t'], aug['cum_inception'],    color='#ffcc00', linewidth=0.8,
                     label='Inception spread')
        axes[2].plot(aug['t'], aug['cum_revaluation'],  color='#cc44ff', linewidth=0.8,
                     label='Inventory revaluation')
        axes[2].axhline(0, color='#444', linewidth=0.6)
        axes[2].set_title('P&L Decomposition: Inception Spread vs Inventory Revaluation (USD)',
                          color='white', fontsize=13)
        axes[2].set_ylabel('USD', color='white')
        axes[2].legend(facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)

        # Panel 4 — Cumulative fees split
        axes[3].plot(aug['t'], aug['cum_fees'], color='#ff4444', linewidth=0.8, label='Total fees')
        axes[3].plot(aug['t'], maker_fees,      color='#ff8800', linewidth=0.8,
                     linestyle='--', label='Maker fees (A)')
        axes[3].plot(aug['t'], hedge_fees,      color='#ff44cc', linewidth=0.8,
                     linestyle=':', label='Hedge fees (B/C)')
        axes[3].set_title('Cumulative Fees (USD)', color='white', fontsize=13)
        axes[3].set_ylabel('USD', color='white')
        axes[3].set_xlabel('Time (s)', color='white')
        axes[3].legend(facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_per_trade_mtm(df: pd.DataFrame, top_n: int = 10) -> None:
        """
        Per-trade MtM evolution chart.

        For each of the top_n MM fills (ranked by fill size), plots how that
        fill's individual MtM contribution evolves over the remainder of the
        session as fair_mid changes. The aggregate session MtM is shown in the
        background for reference.

        Each line starts at the fill's inception time with value
        ≈ inception_spread_i − fee_i, then drifts as mid moves.

        Parameters
        ----------
        df     : Quoter.trade_history DataFrame
        top_n  : number of fills to highlight (default 10)
        """
        if df.empty:
            print("No trades to plot.")
            return

        aug = PnLTracker.augment(df)
        ev  = PnLTracker.per_trade_mtm_evolution(df)

        mm = df[~df['is_hedge']]
        if mm.empty:
            print("No MM fills to plot.")
            return

        top_labels = mm.nlargest(min(top_n, len(mm)), 'size').index.tolist()
        # Map DataFrame label → integer position
        pos_map   = {lbl: pos for pos, lbl in enumerate(df.index)}
        top_pos   = [pos_map[lbl] for lbl in top_labels]

        palette = plt.cm.tab10.colors if top_n <= 10 else plt.cm.tab20.colors
        t_vals  = df['t'].values

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#111111')
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#444444')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#444444')

        # Aggregate MtM as grey background reference
        ax.plot(aug['t'], aug['mtm_pnl'], color='#555555', linewidth=1.2,
                linestyle='--', label='Aggregate MtM', zorder=1)
        ax.axhline(0, color='#444', linewidth=0.6)

        # Per-trade lines
        for k, pos in enumerate(top_pos):
            col   = ev.iloc[:, pos]
            valid = ~col.isna()
            lbl   = top_labels[k]
            size  = df.loc[lbl, 'size']
            dirn  = df.loc[lbl, 'direction']
            color = palette[k % len(palette)]
            ax.plot(t_vals[valid], col[valid], color=color, linewidth=0.9,
                    label=f'Fill {pos} ({dirn}, {size:,.0f} EUR)', zorder=2)
            # Mark inception point
            ax.scatter([t_vals[pos]], [col.iloc[pos]],
                       color=color, s=35, zorder=3)

        ax.set_title(
            f'Per-trade MtM Evolution — Top {len(top_pos)} MM fills by size (USD)',
            color='white', fontsize=13,
        )
        ax.set_ylabel('MtM contribution (USD)', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white',
                  fontsize=8, loc='upper left')

        plt.tight_layout()
        plt.show()
