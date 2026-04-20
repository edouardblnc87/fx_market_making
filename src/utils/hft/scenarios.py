from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum

from .hft_config import HFTConfig


class HFTState(IntEnum):
    ACTIVE        = 0
    ONE_SIDED_BID = 1  # only quoting bids (inventory too short, need to buy)
    ONE_SIDED_ASK = 2  # only quoting asks (inventory too long, need to sell)
    OFFLINE       = 3


@dataclass
class ScheduledEvent:
    start_day:     float
    duration_days: float
    state:         HFTState


# ── Isolated scenario configs (for short 1–5 day diagnostic runs) ────────────
# Each entry is (HFTConfig, schedule). The schedule forces specific state windows;
# between events the natural competitiveness checks apply.

ISOLATED: dict[str, tuple[HFTConfig, list[ScheduledEvent]]] = {
    "normal": (
        HFTConfig(),
        [],
    ),
    "vol_spike": (
        HFTConfig(
            vol_offline_threshold=0.22,    # offline above 22% annualized vol (vs default 35%)
            min_net_half_spread_bps=1.0,   # more selective on profitability
        ),
        [],
    ),
    "hft_onesided": (
        HFTConfig(),
        [
            ScheduledEvent(0.2,  0.25, HFTState.ONE_SIDED_BID),  # day 0.2 → 0.45 (4.8h–10.8h)
            ScheduledEvent(0.6,  0.25, HFTState.ONE_SIDED_ASK),  # day 0.6 → 0.85 (14.4h–20.4h)
        ],
    ),
    "hft_offline": (
        HFTConfig(spread_fraction=999.0),  # spread_fraction so high it never quotes
        [],
    ),
}

# ── Realistic month schedule (for the continuous 30-day run) ─────────────────
# Forced state overrides; between events HFT is ACTIVE (natural triggers still apply).

REALISTIC_MONTH: list[ScheduledEvent] = [
    ScheduledEvent(3.0,  0.5, HFTState.OFFLINE),         # brief outage day 3
    ScheduledEvent(7.0,  1.5, HFTState.ONE_SIDED_BID),   # inventory squeeze day 7–8
    ScheduledEvent(12.0, 3.0, HFTState.OFFLINE),         # major outage days 12–15
    ScheduledEvent(17.0, 2.0, HFTState.ONE_SIDED_ASK),   # reverse squeeze days 17–19
    ScheduledEvent(20.5, 0.3, HFTState.OFFLINE),         # vol spike day 20
    ScheduledEvent(24.0, 2.5, HFTState.ONE_SIDED_BID),   # days 24–26
    ScheduledEvent(26.5, 0.3, HFTState.OFFLINE),         # vol spike day 26
    ScheduledEvent(28.0, 1.5, HFTState.ONE_SIDED_ASK),   # final squeeze days 28–29
]
