# `hft` — Competitor HFT Agent (Phase 3)

Models a latency-advantaged HFT market maker quoting on Exchange A alongside the firm. The HFT sees B/C with **50 ms** lag (vs. 200/170 ms for the MM), so it reprices faster and always undercuts when active — the MM must learn to compete with stale pricing, inventory squeezes, and sudden withdrawals.

---

## Files

| File | Role |
|---|---|
| [hft_agent.py](hft_agent.py) | `HFTAgent` — the step loop, state machine, EWMA vol, fill listener |
| [hft_config.py](hft_config.py) | `HFTConfig` — all tunable parameters (spread, latency, thresholds) |
| [scenarios.py](scenarios.py) | `HFTState`, `ScheduledEvent`, `ISOLATED` dict, `make_realistic_schedule` |

---

## State machine

```
ACTIVE          — quotes both sides, tighter than the MM
ONE_SIDED_BID   — only bids   (ask withdrawn → forces MM to sell against HFT)
ONE_SIDED_ASK   — only asks   (bid withdrawn → forces MM to buy against HFT)
OFFLINE         — no quotes at all
```

Transitions resolved every step in priority order ([hft_agent.py:168](hft_agent.py#L168)):

1. **Schedule override** — any `ScheduledEvent` whose day-window contains `t` forces its state.
2. **Recovery timer** — `OFFLINE → ACTIVE` after `recovery_s` seconds.
3. **Profitability** — if `spread_fraction · spread_AS − fee_bps < min_net_half_spread_bps` → `OFFLINE`.
4. **Vol stress** — if `σ > vol_offline_threshold` (annualised) → `OFFLINE`.
5. **Trend-driven adverse selection** — mid move over `trend_window_s`:
   - move > `+trend_threshold_bps` (uptrend) → ask is stale-cheap → `ONE_SIDED_BID`
   - move < `−trend_threshold_bps` (downtrend) → bid is stale-expensive → `ONE_SIDED_ASK`
6. Else → `ACTIVE`.

`ONE_SIDED_*` is **only** reached via (1) or (5) — never from a pure inventory check.

---

## Step loop (`HFTAgent.step`)

Called once per simulation step, **before** client orders arrive:

1. Cancel all resting HFT orders (`_hft_resting_ids`).
2. Update EWMA variance from `market_B.noised_mid_price[step]` (span = 600 steps).
3. Compute A-S half-spread using fixed internal params (`γ=0.1`, `ω=1/(8·3600)`, `k=0.3`) and current `σ`.
4. Run `_update_state` (state machine above).
5. If `OFFLINE`: return. Otherwise read B/C at step `max(0, step − lag)` where `lag = round(latency_s / dt)` (= 1 step at `dt=50 ms`).
6. Quote `fair_mid ± spread_fraction · half_spread_AS`, snapped to `TICK_SIZE = 0.0001`.
7. Post `buy` and/or `sell` at `max_depth_eur` depending on state.

Fills are received via `on_hft_fill(FillEvent)` registered on `Order_book` (same producer/listener pattern as the MM). The HFT tracks its own `_inventory` and `_fill_history`; exposed as a DataFrame via `hft.fill_history`.

---

## Scenarios

### `ISOLATED` — short diagnostic runs (1–5 days)

| Key | Purpose |
|---|---|
| `normal` | Defaults, no schedule — baseline |
| `vol_spike` | Lower `vol_offline_threshold` (0.28) to trigger frequent vol-stress OFFLINE |
| `hft_onesided` | Two scheduled windows forcing `ONE_SIDED_BID` then `ONE_SIDED_ASK` |
| `hft_offline` | `min_net_half_spread_bps = 1e9` — permanently OFFLINE |

### `REALISTIC_MONTH` / `make_realistic_schedule(n_days)`

Eight scheduled events expressed as **fractions** of `n_days`, so the same pattern (2 squeezes each side, 3 outages, 2 vol spikes) rescales to any run length. Use for full-month backtests.

---

## Parameters (`HFTConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `spread_fraction` | `0.4` | Fraction of A-S half-spread at which HFT quotes. Must stay below `QuoterConfig.stale_tight_fraction` to remain the tightest book. |
| `max_depth_eur` | `5 000` | Size posted per side |
| `latency_s` | `0.050` | B/C lag — 1 step at `dt = 50 ms` |
| `fee_A_maker` | `0.0002` | Maker fee on A, used in profitability check |
| `min_net_half_spread_bps` | `0.3` | Minimum net half-spread to justify quoting |
| `vol_offline_threshold` | `0.12` | Annualised σ above which HFT goes OFFLINE (~2× baseline vol) |
| `recovery_s` | `30.0` | Seconds OFFLINE before auto-returning to ACTIVE |
| `trend_window_s` | `5.0` | Lookback for directional trend detection |
| `trend_threshold_bps` | `0.4` | Mid-move magnitude over the window that triggers ONE_SIDED |

Internal A-S constants (`γ`, `ω`, `k`) are **hard-coded** in `HFTAgent.__init__` to match Phase 1 Quoter defaults. They are intentionally not exposed — the HFT is modelled as a fixed external actor, not a strategy to tune.
