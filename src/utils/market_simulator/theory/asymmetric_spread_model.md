# The Asymmetric Mean-Reverting Spread
## Full Theory and Implementation Plan for B/C Market Simulation

---

## Context

You have already simulated and pre-computed:

- `V[0..N]` — the true asset value path (GBM)
- `P_mid_BC[0..N]` — the B/C mid-price (= V + microstructure noise)

The goal of this document is to go from `P_mid_BC(t)` to `bid_BC(t)` and `ask_BC(t)` using an asymmetric mean-reverting spread that is empirically grounded and realistic.

The universal formula is always:

$$P^{ask}_{BC}(t) = P^{mid}_{BC}(t) + \frac{S(t)}{2}$$

$$P^{bid}_{BC}(t) = P^{mid}_{BC}(t) - \frac{S(t)}{2}$$

**Everything in this document is about how to compute S(t).**

---

## 1. Why Not a Static Spread?

A static spread `S(t) = S_0` assumes volatility is constant. This creates a structural bias: the moments when you most need to hedge on B/C (after large inventory accumulation, which happens during volatile periods) are exactly the moments when B/C spreads are widest. A static spread systematically underestimates your hedging cost during stress.

## 2. Why Not a Pure Vol-Driven Spread?

A purely vol-driven spread `S(t) = S_0 + alpha * RV(t)` is more realistic but produces the wrong shape. During calm periods where `RV(t) ≈ sigma_ann`, the spread is always elevated above `S_0` — it never actually touches its floor. The spread floats permanently above baseline even when nothing is happening, because there is always some realized volatility.

## 3. The Correct Behavior — Version C

What real market makers on liquid venues actually do is empirically well-documented (Engle & Patton 2004, Biais, Hillion & Spatt 1995):

- **Spreads widen fast** during volatility spikes — market makers react defensively and immediately to protect against adverse selection
- **Spreads tighten slowly** after stress — market makers wait to confirm vol has truly normalized before tightening, because tightening too early risks being picked off

This is called **spread asymmetry** or **spread stickiness**. The forces are different in nature:

- Widening is driven by **fear** — the cost of not widening immediately is being adversely selected by informed traders. The incentive to react is immediate and strong.
- Tightening is driven by **competition** — makers only tighten when both confident the stress has passed AND competitive pressure forces them. Both conditions take time.

The resulting behavior:

```
S(t)
|
|         fast spike up
|         /\
|        /  \
|       /    \_____ slow decay back to S_0
S_0 ---/               --------------------------------
|
+-------------------------------------------------------- t
```

---

## 4. The Architecture of S(t)

The spread is always decomposed into two separate components:

$$S(t) = S_0 + S^{excess}(t)$$

Where:
- `S_0` is the **static floor** — the minimum competitive spread, never changes, always present
- `S_excess(t)` is the **dynamic component** — mean-reverts asymmetrically around a vol-driven target

This separation is important. `S_0` embeds the expected adverse selection cost at normal volatility (order processing costs + inventory costs + baseline adverse selection). `S_excess(t)` captures only the **excess** cost above that baseline, driven by volatility surprises.

### The Floor Constraint

`S_excess(t)` is always non-negative:

$$S^{excess}(t) \geq 0 \quad \Rightarrow \quad S(t) \geq S_0$$

The spread can never go below the competitive floor. Without this constraint, during a prolonged calm period the tightening process could push `S_excess` below zero, which is economically nonsensical — competition prevents the spread from going below cost.

---

## 5. The Target Spread

At every moment, there is a **target** that `S_excess(t)` wants to mean-revert toward:

$$S^*(t) = \alpha \cdot RV^{ann}(t)$$

This is the vol-driven level the excess spread *wants* to be at given current volatility. When `RV_ann(t) = 0` the target is zero — `S_excess` wants to be at the floor. When vol spikes, the target jumps up immediately.

The full target spread including the floor:

$$S^{total*}(t) = S_0 + S^*(t) = S_0 + \alpha \cdot RV^{ann}(t)$$

This is where Version 2 (pure stochastic spread) stops. Version C says: **the spread does not jump instantly to its target**. It moves toward it at different speeds depending on direction.

---

## 6. Realized Volatility — The Precise Formula

`RV_ann(t)` is the **annualized rolling standard deviation of log-returns** over the last n steps.

### Step 1 — Log-returns

$$r_j = \ln\frac{V_j}{V_{j-1}}$$

You use `V_t` (the true price path), not `P_mid_BC(t)`. Reason: `P_mid_BC` has microstructure noise on top of `V_t`. Computing volatility from the noised mid would overestimate it by including noise variance. In your simulation you have direct access to `V_t`, so use it.

You use **log-returns** not simple returns because they are:
- The natural measure for GBM (which generated V_t)
- Time-additive: log-return over two periods = sum of individual log-returns
- Approximately symmetric around zero for small moves

### Step 2 — Rolling Realized Variance (zero-mean estimator)

$$RVar(t) = \frac{1}{n} \sum_{k=1}^{n} r^2_{t-k}$$

You use `r^2` not `(r - r_bar)^2`. Reason: with zero drift (mu=0), the zero-mean estimator is both unbiased and has lower variance than the sample variance estimator over short windows. Since you set mu=0 in your GBM, this is the correct choice.

### Step 3 — Per-step to Annualized Scaling

$$RV^{ann}(t) = \sqrt{RVar(t)} \cdot C_{ann}$$

Where the annualization constant is:

$$C_{ann} = \sqrt{\frac{N_{sec/day} \cdot N_{trading\_days}}{dt}} = \sqrt{\frac{23400 \cdot 252}{dt}}$$

With `dt = 0.001`:

$$C_{ann} = \sqrt{\frac{23400 \times 252}{0.001}} = \sqrt{5{,}896{,}800{,}000} \approx 76{,}792$$

After annualization, `RV_ann(t)` is a number around 0.02 (2%) during calm periods — the same scale as your input parameter `sigma_daily = 0.02`. This makes `alpha` interpretable directly in price/vol units.

### The Window n — How to Choose

The window `n` controls the lookback horizon for vol estimation:

| n | Horizon (at dt=1ms) | Behavior |
|---|---|---|
| 1,000 | 1 second | Very reactive, noisy |
| 5,000 | 5 seconds | Good balance |
| 10,000 | 10 seconds | Smooth, slow to react |

Empirical evidence (Engle & Lange 2001) suggests 1–5 second windows best predict spread changes on liquid equities. **Start with n = 5,000.**

### The Lag Effect

`RV_ann(t)` looks backward over n steps. During a sudden vol spike, it takes n steps to fully reflect the new vol level. This is **realistic** — market makers on B/C also observe past volatility and adjust gradually. But it means your spread model has a natural lag of ~n*dt seconds in responding to vol shocks.

---

## 7. The Asymmetric Mean Reversion — Core Mechanism

The dynamic component `S_excess(t)` evolves as:

$$dS^{excess}(t) = \kappa(t) \cdot \left(S^*(t) - S^{excess}(t)\right) dt$$

Where `kappa(t)` is the **speed of mean reversion** — and this is where the asymmetry lives:

$$\kappa(t) = \begin{cases} \kappa_u & \text{if } S^*(t) > S^{excess}(t) \quad \text{(target above current} \Rightarrow \text{widening)} \\ \kappa_d & \text{if } S^*(t) \leq S^{excess}(t) \quad \text{(target below current} \Rightarrow \text{tightening)} \end{cases}$$

With **kappa_u >> kappa_d**: upward speed is much faster than downward speed.

This is a **regime-switching mean reversion** process. The regime is determined entirely by whether the target is above or below the current excess spread.

### Economic Interpretation of kappa

`kappa` is a **reversion speed** measured in units of (1/time). The **half-life** — time for the gap between current and target to halve — is:

$$t_{1/2} = \frac{\ln 2}{\kappa}$$

This is the most intuitive way to set your parameters:

| Parameter | Half-life | Meaning |
|---|---|---|
| `kappa_u = 50` | ~14ms | Spread reaches target in ~14ms after vol spike |
| `kappa_d = 2` | ~350ms | Spread takes ~350ms to normalize after stress |

The ratio `kappa_u / kappa_d = 25` encodes that **widening is 25x faster than tightening**.

---

## 8. Discrete-Time Implementation

In discrete time with step `dt`, the evolution becomes:

$$S^{excess}(t+dt) = \underbrace{(1 - \kappa(t) \cdot dt)}_{\text{weight on current}} \cdot S^{excess}(t) + \underbrace{\kappa(t) \cdot dt}_{\text{weight on target}} \cdot S^*(t)$$

This is a **weighted average** between the current excess spread and the target.

### Stability Condition

For the discrete process to be stable (no overshooting):

$$\kappa(t) \cdot dt < 1$$

With `dt = 0.001s`:
- Requires `kappa < 1000` per second
- Your values (`kappa_u = 50`, `kappa_d = 2`) satisfy this with large margin

If `kappa * dt = 1`, the spread jumps instantly to target (infinite speed). If `kappa * dt = 0`, the spread never moves. Your values give smooth, well-behaved dynamics.

### Floor Re-enforcement

After each update, enforce:

$$S^{excess}(t+dt) = \max\left(S^{excess}(t+dt),\ 0\right)$$

This prevents `S_excess` from going negative during prolonged calm periods.

---

## 9. Optional: Adding Spread Noise

In reality the spread doesn't follow a perfectly smooth mean-reverting path. Individual order cancellations and insertions create small random fluctuations. You can add a noise term:

$$S^{excess}(t+dt) = (1 - \kappa(t) \cdot dt) \cdot S^{excess}(t) + \kappa(t) \cdot dt \cdot S^*(t) + \sigma_S \cdot \sqrt{dt} \cdot \varepsilon^S_t$$

Where:
- `eps_S_t ~ N(0,1)` independent of all other random processes
- `sigma_S` = spread noise volatility, set small: `sigma_S ≈ 0.1 * alpha * sigma_step`

After adding noise, re-enforce the floor. This is optional for Phase 1 but makes the spread path look more realistic (slightly ragged around its mean-reverting path).

---

## 10. The Complete Computation Pipeline

All steps are done **before the main simulation loop**. The result is a set of pre-computed arrays that you read from inside the loop.

### Step 1 — Log-Returns
From `V[0..N]`, compute `r[j] = ln(V[j] / V[j-1])` for all j.

### Step 2 — Rolling Realized Variance
For each `t > n`, compute `RVar[t] = mean(r[t-n:t]^2)` — the mean squared return over the last n steps.
For `t <= n`, pad with the theoretical variance: `RVar[t] = sigma_step^2`.

### Step 3 — Annualize
`RV_ann[t] = sqrt(RVar[t]) * C_ann` where `C_ann = sqrt(23400 * 252 / dt)`.

### Step 4 — Compute Target Excess Spread
`S_star[t] = alpha * RV_ann[t]`

### Step 5 — Evolve S_excess Forward (Sequential — Cannot Be Vectorized)
Initialize: `S_excess[0] = 0`

For each `t` from 0 to N-1:
```
if S_star[t] > S_excess[t]:
    kappa = kappa_u          # widening regime
else:
    kappa = kappa_d          # tightening regime

S_excess[t+1] = (1 - kappa*dt) * S_excess[t] + kappa*dt * S_star[t]
S_excess[t+1] = max(S_excess[t+1], 0)   # floor enforcement
```

This step is the **only sequential step** in the B/C simulation. All other steps are vectorizable.

### Step 6 — Full Spread
`S[t] = S_0 + S_excess[t]`

### Step 7 — Bid and Ask
```
ask_BC[t] = P_mid_BC[t] + S[t] / 2
bid_BC[t] = P_mid_BC[t] - S[t] / 2
```

---

## 11. What Market A Consumes From This

### Every Timestep

Market A reads the **delayed** mid-price to center its quotes on exchange A:

$$\hat{P}(t) = P^{mid}_{BC}[t - \Delta_{you\_steps}]$$

Market A also reads the **delayed** spread to feed into its Ho-Stoll formula for spread width:

$$\hat{S}(t) = S[t - \Delta_{you\_steps}]$$

where `Delta_you_steps = int(0.200 / dt)` (200ms expressed in steps).

### Only at Hedge Events

When Market A's inventory hits the 90% threshold, it executes a hedge on B/C. At that moment it needs the **current** (not delayed) bid/ask — because the hedge executes at the current market price:

- If `I_t > 0` (long, needs to sell on B/C): execute at `bid_BC[i]`
- If `I_t < 0` (short, needs to buy on B/C): execute at `ask_BC[i]`

**The asymmetry is critical:** for quoting on A you use the delayed observation. For hedge execution you use the current price. The hedging cost is:

$$\text{Hedge cost per unit} = \frac{S[i]}{2} + \text{fees}$$

With the asymmetric spread, this cost is now time-varying — higher during volatile periods when you most need to hedge. This is the key realism that Version 1 (static spread) missed.

### HFT Adverse Selection

HFTs observe B/C with only 50ms delay:

$$\hat{P}_{HFT}(t) = P^{mid}_{BC}[t - \Delta_{HFT\_steps}]$$

where `Delta_HFT_steps = int(0.050 / dt)`. They also see the spread earlier. During a vol spike, the spread on B/C widens fast (kappa_u is large). HFTs see this widening 150ms before you. They can exploit the window where the B/C spread has already widened (informing them the market is stressed) but your quotes on A haven't yet updated.

---

## 12. Alpha Calibration — The Practical Rule

You want: during a 2-sigma volatility event, the B/C spread approximately doubles.

A 2-sigma event means:

$$RV^{ann}_{stress} = 2 \cdot \sigma_{ann}$$

You want:

$$S_0 + \alpha \cdot 2\sigma_{ann} = 2 \cdot S_0$$

Solving:

$$\boxed{\alpha = \frac{S_0}{2 \cdot \sigma_{ann}}}$$

Example with `S_0 = 0.01` and `sigma_daily = 0.02`, `sigma_ann = 0.02 * sqrt(252) ≈ 0.317`:

$$\alpha = \frac{0.01}{2 \times 0.317} \approx 0.016 \text{ dollars per unit annualized vol}$$

Interpretation: a 1% increase in annualized vol widens the spread by 1.6 cents.

---

## 13. Full Parameter Reference

| Parameter | Symbol | Meaning | Starting Value |
|---|---|---|---|
| Static floor | `S_0` | Minimum competitive spread | 0.01 (1 tick) |
| Vol sensitivity | `alpha` | Spread widening per unit ann. vol | `S_0 / (2 * sigma_ann)` |
| RV window | `n` | Lookback steps for realized vol | 5000 (5s at 1ms) |
| Upward speed | `kappa_u` | Mean reversion speed when widening | 50 per second |
| Downward speed | `kappa_d` | Mean reversion speed when tightening | 2 per second |
| Spread noise | `sigma_S` | Optional noise on spread path | `0.1 * alpha * sigma_step` |
| Annualization | `C_ann` | Scaling factor for RV | `sqrt(23400 * 252 / dt)` |

---

## 14. Behavior Produced

```
Vol spike at t = t_0:

S_star(t)  ______/\__________/\_______
                              (second spike)

S_excess   ______/  \________/  \______
                fast up  slow down

S(t)       S_0 + S_excess:
           ___/  \__________/  \_______ S_0
           never below S_0
           spikes fast, decays slowly
           calm periods → S(t) → S_0
```

During calm periods `S_excess → 0` exponentially at rate `kappa_d`, so `S(t) → S_0`. This is the correct floor behavior: in equilibrium, competition brings the spread back to the minimum cost level.

---

## 15. Literature Sources

| Result | Source |
|---|---|
| Mid-price = efficient price + noise | Roll (1984), *Journal of Finance* |
| Competitive equilibrium spread | Demsetz (1968), *Quarterly Journal of Economics* |
| Spread widens with adverse selection | Glosten & Milgrom (1985), *Journal of Financial Economics* |
| Spread components estimation | Glosten & Harris (1988), *Journal of Financial Economics* |
| Spread asymmetry and stickiness | Engle & Patton (2004), *Journal of Financial Markets* |
| Intraday spread patterns | Admati & Pfleiderer (1988), *Review of Financial Studies* |
| Realized volatility and spread | Engle & Lange (2001), *Journal of Financial Econometrics* |
| Empirical limit order book | Biais, Hillion & Spatt (1995), *Journal of Finance* |

---

*Document prepared for Market Making Project — Phase 1 World Simulation*
