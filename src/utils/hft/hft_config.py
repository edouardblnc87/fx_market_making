from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HFTConfig:
    # Quoting — spread_fraction must stay below QuoterConfig.stale_tight_fraction
    # so HFTs are always tighter than even our most aggressive Priority-6 reprice.
    spread_fraction:          float = 0.4     # fraction of A-S half_spread
    max_depth_eur:            float = 5_000.0
    latency_s:                float = 0.050   # 50ms → 1 step at dt=50ms

    # Profitability check — global OFFLINE trigger
    # Both sides unprofitable → OFFLINE.
    # ONE_SIDED is exclusively schedule/trend-driven (see hft_agent._update_state).
    fee_A_maker:              float = 0.0002  # maker fee on A
    min_net_half_spread_bps:  float = 0.3     # min net profit per side to justify quoting

    # Secondary OFFLINE trigger — extreme vol stress only
    vol_offline_threshold:    float = 0.35    # annualised vol → OFFLINE
    recovery_s:               float = 30.0    # seconds offline before returning ACTIVE

    # ONE_SIDED trigger — adverse selection from stale pricing
    # Uptrend → ask is stale-cheap → remove ask → ONE_SIDED_BID
    # Downtrend → bid is stale-expensive → remove bid → ONE_SIDED_ASK
    trend_window_s:           float = 5.0     # lookback window (~1 stdev move at vol=20%)
    trend_threshold_bps:      float = 1.5     # move size to trigger one-sided
