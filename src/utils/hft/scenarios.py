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
            vol_offline_threshold=0.28,    # offline above 28% annualized vol (vs default 35%)
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
        HFTConfig(min_net_half_spread_bps=1e9),  # profitability check always fails → always OFFLINE
        [],
    ),
}

# ── Realistic schedule — scales to any n_days ────────────────────────────────
# Events defined as fractions of total period so frequency is preserved
# regardless of simulation length.  Call make_realistic_schedule(n_days).

_SCHEDULE_TEMPLATE: list[tuple[float, float, HFTState]] = [
    # (start_fraction, duration_fraction, state)
    (0.10, 0.017, HFTState.OFFLINE),         # brief outage ~10% through
    (0.23, 0.050, HFTState.ONE_SIDED_BID),   # inventory squeeze ~23–28%
    (0.40, 0.100, HFTState.OFFLINE),         # major outage ~40–50%
    (0.57, 0.067, HFTState.ONE_SIDED_ASK),   # reverse squeeze ~57–64%
    (0.68, 0.010, HFTState.OFFLINE),         # vol spike ~68%
    (0.80, 0.083, HFTState.ONE_SIDED_BID),   # squeeze ~80–88%
    (0.88, 0.010, HFTState.OFFLINE),         # vol spike ~88%
    (0.93, 0.050, HFTState.ONE_SIDED_ASK),   # final squeeze ~93–98%
]


def make_realistic_schedule(n_days: float = 30.0) -> list[ScheduledEvent]:
    """Return a schedule scaled to n_days, preserving relative timing and durations."""
    return [
        ScheduledEvent(frac * n_days, dur * n_days, state)
        for frac, dur, state in _SCHEDULE_TEMPLATE
    ]


# Convenience alias for the classic 30-day run
REALISTIC_MONTH: list[ScheduledEvent] = make_realistic_schedule(30.0)
