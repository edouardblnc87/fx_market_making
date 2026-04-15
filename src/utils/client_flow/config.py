# ---------------------------------------------------------------------------
# Client order flow — default parameters
# Based on Avellaneda & Stoikov (2006), Sections 2.4–2.5
# ---------------------------------------------------------------------------

# Arrival intensity: λ(δ) = A · exp(-k · δ)   where δ is in basis points
# A = baseline arrival rate at δ=0 (per second)
# k = exponential decay rate per bp — higher k means intensity drops faster
#     with distance from mid.  Half-life = ln(2)/k bps.
# The paper uses A = Λ/α = 140, k = αK = 1.5 with δ in dollars (s≈100).
# For FX (EUR/USD ≈ 1.10) we express δ in bps so k has the same
# interpretive meaning: k=1.5 → half-life ≈ 0.46 bp.
A_BUY  = 1.5         # buy-side arrival rate (per second at δ=0)
A_SELL = 1.5          # sell-side arrival rate (can differ for asymmetry)
K_BUY  = 1.5          # buy decay rate per bp  (matches QuoterConfig.k)
K_SELL = 1.5          # sell decay rate per bp

# Power-law size distribution: f(x) ∝ x^(-1-α)   (paper eq. 2.8)
ALPHA    = 1.5        # tail exponent (Gabaix et al.: 1.5, Gopikrishnan: 1.53)
SIZE_MIN = 1_000      # minimum order size in EUR — truncation floor
SIZE_MAX = 100_000    # maximum order size in EUR — truncation ceiling

# Order type mix
MARKET_ORDER_RATIO = 0.5   # 50 % market orders, 50 % limit orders
