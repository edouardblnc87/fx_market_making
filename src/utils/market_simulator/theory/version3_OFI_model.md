# Version 3 — Order Flow Imbalance and Asymmetric Spread
## Full Theory and Implementation Plan for B/C Market Simulation

---

## Context

You have already built:

- `V[0..N]` — true asset value path (GBM)
- `P_mid_BC[0..N]` — B/C mid-price (= V + microstructure noise)
- `S[0..N]` — asymmetric mean-reverting spread (Version C)
- `ask_BC[0..N]`, `bid_BC[0..N]` — symmetric quotes around mid

Version 3 breaks the symmetry assumption. The bid and ask are no longer symmetric around `P_mid_BC`. They are centered on a **flow-adjusted mid** that reflects current directional pressure, and the two half-spreads can differ in response to that pressure.

---

## 1. Why Symmetry Is Wrong

The symmetric spread formula:

$$P^{ask} = P^{mid} + \frac{S}{2} \qquad P^{bid} = P^{mid} - \frac{S}{2}$$

implicitly assumes the market maker has no directional view and no inventory pressure — they are equally happy to buy or sell at any moment. This is only true when order flow is perfectly balanced.

In reality, at any given moment there is directional pressure — more aggressive buyers than sellers, or vice versa. This pressure does two things simultaneously:

**Effect 1 — Shifts the center of the spread.** When there is persistent buy pressure, the entire book shifts upward. Sellers demand more, buyers compete for priority. The quoted mid is no longer the true mid.

**Effect 2 — Widens one side more than the other.** Buy pressure widens the ask (sellers are more reluctant, knowing buyers are eager) while the bid may tighten (buyers compete). The two half-spreads become unequal.

Both effects are documented empirically in Cont, Kukanov & Stoikov (2014) and Hasbrouck & Seppi (2001).

---

## 2. Order Flow Imbalance — What It Is

Order flow imbalance (OFI) is the **net directional pressure** on the market at time t:

$$OFI(t) = \lambda^{buy}(t) - \lambda^{sell}(t)$$

Where `lambda_buy` and `lambda_sell` are the rates of buyer-initiated and seller-initiated market orders hitting the B/C book.

- `OFI > 0` → more buyers than sellers → upward price pressure
- `OFI < 0` → more sellers than buyers → downward price pressure
- `OFI = 0` → balanced flow → symmetric book (reduces to Version 2)

### Empirical Properties of OFI (Cont et al. 2014)

From empirical analysis of high-frequency data:

1. **Strong short-term autocorrelation** — OFI is persistent over milliseconds to seconds
2. **Rapid exponential decay** — autocorrelation dies out quickly, OFI is not persistent over minutes
3. **Zero unconditional mean** — the market is balanced on average
4. **Stationary distribution** — OFI fluctuates within a bounded range, does not trend

These four properties uniquely motivate the Ornstein-Uhlenbeck process as the correct model.

### Price Predictability (Hasbrouck 1991, Cont et al. 2014)

The most important empirical finding: OFI predicts short-term price changes.

$$\mathbb{E}[\Delta P^{mid}(t + \tau) \mid OFI(t)] = \beta \cdot OFI(t) \cdot e^{-\theta\tau}$$

The predictability decays exponentially with horizon `tau` at rate `theta`. At very short horizons, OFI explains 65-80% of mid-price variance (Cont et al. 2014, R² result). This is one of the strongest predictive relationships in market microstructure.

---

## 3. Simulating OFI — The Ornstein-Uhlenbeck Process

You do not simulate individual orders on B/C (that would require Approach 2 — full order book simulation, which is overkill). Instead you model OFI **directly** as a mean-reverting stochastic process whose statistical properties match the empirical findings above.

The Ornstein-Uhlenbeck process:

$$dOFI_t = -\theta \cdot OFI_t \, dt + \sigma_{OFI} \cdot dZ_t$$

In discrete time with step `dt`:

$$OFI_{t+dt} = OFI_t - \theta \cdot OFI_t \cdot dt + \sigma_{OFI} \cdot \sqrt{dt} \cdot \varepsilon^{OFI}_t$$

$$= (1 - \theta \cdot dt) \cdot OFI_t + \sigma_{OFI} \cdot \sqrt{dt} \cdot \varepsilon^{OFI}_t$$

Where:
- `theta` = mean reversion speed (per second)
- `sigma_OFI` = volatility of the imbalance process
- `eps_OFI_t ~ N(0,1)` independent of all other random processes
- Initial condition: `OFI_0 = 0` (market starts balanced)

### Why OU Is The Right Model

| Empirical property | OU property |
|---|---|
| Zero unconditional mean | OU mean-reverts to zero by construction |
| Stationary distribution | OU has stationary distribution `N(0, sigma_OFI^2 / (2*theta))` |
| Exponential autocorrelation decay | `Corr(OFI_t, OFI_{t+tau}) = exp(-theta * tau)` exactly |
| Short-term persistence | Captured by `1/theta` = characteristic timescale |

No other simple continuous-time process simultaneously satisfies all four properties.

### The Stationary Distribution

The long-run distribution of `OFI_t` is:

$$OFI_\infty \sim \mathcal{N}\!\left(0,\ \frac{\sigma^2_{OFI}}{2\theta}\right)$$

The standard deviation of OFI in steady state is:

$$\sigma_{OFI}^{stationary} = \frac{\sigma_{OFI}}{\sqrt{2\theta}}$$

This tells you what "typical" OFI looks like. You should set `sigma_OFI` and `theta` together so that this stationary standard deviation matches your intuition about typical imbalance magnitude.

### The Half-Life

The time for OFI autocorrelation to halve:

$$t_{1/2} = \frac{\ln 2}{\theta}$$

For liquid equities, empirical half-lives are 10-100ms (Cont et al. 2014). This corresponds to:
- `theta = 70` per second → half-life ≈ 10ms
- `theta = 7` per second → half-life ≈ 100ms

Start with `theta = 10` per second → half-life ≈ 70ms. Realistic for a moderately liquid market.

### Pre-computation

The entire OFI path `OFI[0..N]` is pre-computed before the loop. It is independent of your strategy and independent of `V_t` (in sub-option 3a — see below). It is a static array you read from inside the loop.

---

## 4. Two Sub-Options — Critical Design Decision

You must decide whether OFI feeds back into `V_t` or not.

### Sub-Option 3a — OFI Affects Quotes Only (Recommended Starting Point)

OFI shifts the bid/ask on B/C but does **not** feed back into `V_t`. The price path is still pure GBM. OFI is purely a quote adjustment signal.

$$V_t: \text{ pure GBM, unchanged}$$
$$\tilde{P}^{mid}_{BC}(t) = P^{mid}_{BC}(t) + \phi \cdot OFI(t)$$

The skew `phi * OFI(t)` is temporary — it mean-reverts as OFI does. When `OFI = 0`, the model collapses exactly to Version 2.

**Use this first.** It keeps `V_t` clean and isolated. The OFI effect is transparent and easy to debug. You can turn it on and off with one parameter `phi`.

### Sub-Option 3b — OFI Has Permanent Price Impact (Hasbrouck 1991)

OFI causes `V_t` itself to drift. Sustained buy pressure actually moves the true value up permanently.

$$dV_t = \beta \cdot OFI_t \cdot dt + \sigma \cdot dW_t$$

In discrete time:

$$V_{t+dt} = V_t \cdot \exp\!\left(-\frac{\sigma^2_{step}}{2} + \sigma_{step} \cdot \varepsilon_t + \beta \cdot OFI_t \cdot dt\right)$$

The `beta * OFI_t * dt` term adds a drift to `V_t` proportional to the current imbalance.

**This is more realistic but more complex.** It creates a coupled system: OFI drifts V, V moves, RV increases, spread widens, which in turn affects your adverse selection on A. The three-way interaction produces volatility clustering — periods of high vol and high OFI cluster together, which is empirically observed.

**Use this in Phase 2+ once 3a is working correctly.**

---

## 5. The Two Effects on Bid/Ask — Full Model

As established in Section 1, OFI produces two separate effects on the quotes. You model them independently.

### Effect 1 — The Skewed Mid (Center Shift)

Define the **flow-adjusted mid**:

$$\tilde{P}^{mid}_{BC}(t) = P^{mid}_{BC}(t) + \phi \cdot OFI(t)$$

Where `phi > 0` is the sensitivity of the mid to order flow. When `OFI > 0` (buy pressure), the center of the book shifts up. When `OFI < 0` (sell pressure), it shifts down.

`phi` has units of **price per unit OFI**. It should be calibrated so that a one-standard-deviation OFI event shifts the mid by approximately half a tick:

$$\phi \cdot \sigma_{OFI}^{stationary} \approx \frac{\text{tick}}{2}$$

$$\phi \approx \frac{\text{tick}}{2 \cdot \sigma_{OFI}^{stationary}} = \frac{\text{tick} \cdot \sqrt{2\theta}}{2 \cdot \sigma_{OFI}}$$

### Effect 2 — The Asymmetric Half-Spreads

The two sides of the book widen asymmetrically in the direction of flow:

$$\delta^{ask}(t) = \frac{S(t)}{2} + \psi \cdot \max(OFI(t),\ 0)$$

$$\delta^{bid}(t) = \frac{S(t)}{2} + \psi \cdot \max(-OFI(t),\ 0)$$

Where `psi > 0` is the asymmetric widening parameter.

**Intuition:**
- `OFI > 0` (buy pressure): ask widens by `psi * OFI`, bid stays at `S/2` — sellers demand a premium when buyers are aggressive
- `OFI < 0` (sell pressure): bid widens by `psi * |OFI|`, ask stays at `S/2` — buyers demand a discount when sellers are aggressive
- `OFI = 0`: both half-spreads equal `S/2` — reduces to symmetric Version 2

Note that `max(OFI, 0)` and `max(-OFI, 0)` are never both positive simultaneously. At most one side widens at any given moment.

### The Total Spread Width

The total spread is now:

$$S^{total}(t) = \delta^{ask}(t) + \delta^{bid}(t) = S(t) + \psi \cdot |OFI(t)|$$

The spread widens with the **magnitude** of imbalance, regardless of direction. This makes sense: any strong directional pressure — buy or sell — increases adverse selection risk for the side being hit.

---

## 6. The Complete Bid/Ask Formula

Combining both effects:

$$P^{ask}_{BC}(t) = \tilde{P}^{mid}_{BC}(t) + \delta^{ask}(t)$$
$$= P^{mid}_{BC}(t) + \phi \cdot OFI(t) + \frac{S(t)}{2} + \psi \cdot \max(OFI(t),\ 0)$$

$$P^{bid}_{BC}(t) = \tilde{P}^{mid}_{BC}(t) - \delta^{bid}(t)$$
$$= P^{mid}_{BC}(t) + \phi \cdot OFI(t) - \frac{S(t)}{2} - \psi \cdot \max(-OFI(t),\ 0)$$

### Verification — Symmetry When OFI = 0

When `OFI = 0`:

$$P^{ask}_{BC} = P^{mid}_{BC} + \frac{S(t)}{2}$$
$$P^{bid}_{BC} = P^{mid}_{BC} - \frac{S(t)}{2}$$

Exactly Version 2. The model collapses correctly. ✅

### The Quoted Mid vs True Mid

The **quoted mid** — the midpoint between bid and ask as observed — is:

$$P^{quoted\_mid}(t) = \frac{P^{ask}_{BC}(t) + P^{bid}_{BC}(t)}{2}$$

With asymmetric half-spreads, this no longer equals `P_mid_BC(t)`. The difference is:

$$P^{quoted\_mid}(t) - P^{mid}_{BC}(t) = \phi \cdot OFI(t) + \frac{\psi}{2}\left(\max(OFI,0) - \max(-OFI,0)\right)$$
$$= \phi \cdot OFI(t) + \frac{\psi}{2} \cdot OFI(t)$$
$$= \left(\phi + \frac{\psi}{2}\right) \cdot OFI(t)$$

The quoted mid deviates from the true mid by a factor proportional to OFI. This is empirically documented — the quoted mid leads the efficient price during periods of order flow imbalance (Hasbrouck 1991).

---

## 7. Your Delayed Observation — The Adversity of Latency

You observe B/C with a 200ms delay. HFTs observe with 50ms delay. The OFI signal has its own latency penalty that compounds the price staleness problem.

### OFI Autocorrelation Decay

The correlation between what you observe and current OFI:

$$\text{Corr}(\widehat{OFI}_{you}(t),\ OFI(t)) = e^{-\theta \cdot \Delta_{you}}$$

For `theta = 10` per second, `Delta_you = 0.200s`:

$$e^{-10 \times 0.200} = e^{-2} \approx 0.135$$

Your observed OFI has only **13.5% correlation** with the current true OFI. It is nearly useless as a predictive signal.

For HFTs with `Delta_HFT = 0.050s`:

$$e^{-10 \times 0.050} = e^{-0.5} \approx 0.607$$

HFTs have **60.7% correlation** with current OFI. Their signal is meaningful.

### The Practical Implication

This mathematical result is important for your project: **your latency disadvantage is even more severe for OFI than for price**. The 150ms gap destroys OFI predictability almost completely. HFTs can exploit OFI profitably. You largely cannot.

When `OFI > 0` (buy pressure), HFTs:
1. See it 150ms before you
2. Know the ask on B/C has already been pushed up (Effect 1 + Effect 2)
3. Know the true mid has already moved up (if sub-option 3b)
4. Hit your stale ask on A before you can reprice

You observe the OFI signal only after the price has already moved and HFTs have already traded against you. Adding OFI to your strategy in Phase 2 will show limited benefit — which is the correct and realistic result.

---

## 8. The Complete Pre-Loop Computation Pipeline

All steps are done before the main simulation loop.

### Step 1 — Simulate OFI Path

Initialize `OFI[0] = 0`. For each step t:

$$OFI[t+1] = (1 - \theta \cdot dt) \cdot OFI[t] + \sigma_{OFI} \cdot \sqrt{dt} \cdot \varepsilon^{OFI}_t$$

This is the only new array you need to simulate. It is independent of `V_t` in sub-option 3a.

### Step 2 — (Sub-option 3b only) Modify V_t Path

If using permanent price impact, re-generate `V_t` incorporating OFI drift:

$$V[t+1] = V[t] \cdot \exp\!\left(-\frac{\sigma^2_{step}}{2} + \sigma_{step} \cdot \varepsilon_t + \beta \cdot OFI[t] \cdot dt\right)$$

Otherwise skip this step.

### Step 3 — Compute P_mid_BC (unchanged from before)

$$P^{mid}_{BC}[t] = V[t] + \eta_t$$

### Step 4 — Compute S(t) (unchanged from Version C)

Follow the full asymmetric mean-reverting spread pipeline:
- Compute log-returns from V
- Compute rolling realized variance
- Annualize to RV_ann
- Evolve S_excess with asymmetric kappa_u, kappa_d
- S[t] = S_0 + S_excess[t]

### Step 5 — Compute Flow-Adjusted Mid

$$\tilde{P}^{mid}[t] = P^{mid}_{BC}[t] + \phi \cdot OFI[t]$$

### Step 6 — Compute Asymmetric Half-Spreads

$$\delta^{ask}[t] = \frac{S[t]}{2} + \psi \cdot \max(OFI[t],\ 0)$$

$$\delta^{bid}[t] = \frac{S[t]}{2} + \psi \cdot \max(-OFI[t],\ 0)$$

### Step 7 — Compute Bid and Ask

$$ask_{BC}[t] = \tilde{P}^{mid}[t] + \delta^{ask}[t]$$

$$bid_{BC}[t] = \tilde{P}^{mid}[t] - \delta^{bid}[t]$$

**Result:** Six pre-computed arrays available for the main loop:

```
OFI[0..N]           raw OFI path
P_mid_BC[0..N]      mid-price
S[0..N]             spread width (from Version C)
P_skewed[0..N]      flow-adjusted mid
ask_BC[0..N]        full ask quotes
bid_BC[0..N]        full bid quotes
```

---

## 9. What Market A Consumes

### Every Timestep — Delayed Observation

Market A reads arrays at index `i - delay_steps_you`:

```
P_hat(t)     = P_mid_BC[i - delay_you]    center of your quotes on A
OFI_hat(t)   = OFI[i - delay_you]         weak directional signal (13.5% corr)
S_hat(t)     = S[i - delay_you]           spread estimate for Ho-Stoll input
```

### Quote Skewing on A Using OFI

Your delayed OFI feeds into your quote skewing on A as an additional term beyond inventory:

$$\tilde{P}^m_A(t) = \hat{V}(t) - \gamma \cdot I_t - \psi_A \cdot \widehat{OFI}_{you}(t)$$

Where:
- `gamma * I_t` = inventory skew (Ho-Stoll)
- `psi_A * OFI_hat` = flow skew (weak but non-zero signal)

Given the 13.5% correlation, `psi_A` should be small. The signal is weak but not zero — you use what you have.

### Only at Hedge Events — Current Prices

When hedging, Market A crosses the spread on B/C. With asymmetric spreads, the cost is now directionally dependent:

**If `I_t > 0` (long, selling on B/C):**
You hit the bid. The cost is `delta_bid[i]` not `S[i]/2`.

$$\text{Hedge cost} = \delta^{bid}[i] = \frac{S[i]}{2} + \psi \cdot \max(-OFI[i],\ 0)$$

If there is sell pressure (`OFI < 0`) when you need to sell on B/C, your cost is higher than half the spread. You are selling into a market that is also under sell pressure — exactly the worst time, and the model correctly penalizes it.

**If `I_t < 0` (short, buying on B/C):**
You hit the ask. The cost is `delta_ask[i]`.

$$\text{Hedge cost} = \delta^{ask}[i] = \frac{S[i]}{2} + \psi \cdot \max(OFI[i],\ 0)$$

If there is buy pressure (`OFI > 0`) when you need to buy on B/C, your cost is higher. Again: worst-case timing, correctly penalized.

### The Adverse Timing Effect

In practice, your inventory accumulates in the direction of persistent order flow:

```
OFI > 0 (buy pressure on B/C)
    → informed buyers hit your ask on A
    → your inventory: I_t goes negative (you've sold)
    → you need to BUY on B/C to hedge
    → BUT buy pressure means ask_BC is elevated (Effect 2)
    → your hedging cost is higher than normal
```

This double adverse effect — inventory accumulates AND hedge cost rises simultaneously — is a realistic and important feature of the model. It captures why HFT adverse selection is so damaging during trending markets.

---

## 10. Parameter Reference

| Parameter | Symbol | Meaning | Starting Value |
|---|---|---|---|
| OFI mean reversion | `theta` | Speed of OFI decay | 10 per second |
| OFI volatility | `sigma_OFI` | Magnitude of OFI fluctuations | Calibrate to tick/2 |
| Mid shift sensitivity | `phi` | Price per unit OFI | `tick / (2 * sigma_OFI_stationary)` |
| Asymmetric widening | `psi` | Extra half-spread per unit OFI | Small, calibrate |
| Permanent impact | `beta` | OFI drift on V_t (sub-option 3b only) | Start at 0 (3a) |
| OFI stationary std | `sigma_OFI_stat` | `sigma_OFI / sqrt(2*theta)` | Derived parameter |

### Calibrating sigma_OFI

Set `sigma_OFI` so that the stationary standard deviation of OFI produces a typical mid shift of half a tick:

$$\phi \cdot \sigma_{OFI}^{stationary} = \frac{\text{tick}}{2}$$

Combined with the formula for `phi`:

$$\sigma_{OFI}^{stationary} = \frac{\sigma_{OFI}}{\sqrt{2\theta}} = \frac{\text{tick}}{2\phi}$$

You have one free parameter. Set `phi` to a reasonable value (e.g. `phi = 1.0`), then solve for `sigma_OFI`:

$$\sigma_{OFI} = \phi \cdot \frac{\text{tick}}{2} \cdot \sqrt{2\theta}$$

---

## 11. Behavior Produced

```
OFI path (OU process):
     +2  ___     ___
        /   \   /   \
  0  --/-----\_/-----\----  (mean-reverts to zero)
                       \___
     -2

Effect on quotes (OFI > 0 episode):

P_mid_BC:  ───────────────── (unchanged, pure signal)

P_skewed:  ────────╱╲─────── (shifted up during buy pressure)

ask_BC:    ────────╱╲─────── (shifted up + widened)

bid_BC:    ────────╱╲─────── (shifted up, not widened)

Spread:
delta_ask  ────────╱╲─────── (S/2 + psi*OFI during buy pressure)
delta_bid  ─────────────────  (stays at S/2)
```

---

## 12. Comparison With Previous Versions

| Feature | Version 1 | Version 2 | Version C | Version 3 |
|---|---|---|---|---|
| Spread width | Constant | Vol-driven | Asymmetric MR | Asymmetric MR |
| Spread center | P_mid | P_mid | P_mid | P_mid + phi*OFI |
| Half-spread symmetry | ✅ Symmetric | ✅ Symmetric | ✅ Symmetric | ❌ Asymmetric |
| OFI signal | ❌ None | ❌ None | ❌ None | ✅ Present |
| Hedge cost | Fixed | Vol-varying | Vol-varying + sticky | Directional + vol |
| Adverse timing | ❌ Not modeled | ❌ Not modeled | ❌ Not modeled | ✅ Modeled |
| HFT advantage channel | Price only | Price + vol | Price + vol | Price + vol + OFI |

Each version is a strict extension. Setting `phi = 0` and `psi = 0` collapses Version 3 exactly to Version C.

---

## 13. Sources

| Result | Source |
|---|---|
| OFI predicts price changes | Cont, Kukanov & Stoikov (2014), *Journal of Financial Econometrics* |
| Permanent price impact of order flow | Hasbrouck (1991), *Journal of Finance* |
| Common factors in order flow | Hasbrouck & Seppi (2001), *Journal of Financial Economics* |
| Asymmetric spread response to flow | Glosten & Milgrom (1985), *Journal of Financial Economics* |
| Quoted mid leads efficient price | Hasbrouck (1991), *Journal of Finance* |
| Exponential OFI autocorrelation | Cont et al. (2014), empirical result |
| Spread widens with imbalance magnitude | Madhavan, Richardson & Roomans (1997), *Review of Financial Studies* |

---

*Document prepared for Market Making Project — Phase 1/2 World Simulation*
