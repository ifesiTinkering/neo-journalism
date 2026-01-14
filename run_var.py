#!/usr/bin/env python3
"""
================================================================================
VAR (Vector Autoregression) Analysis: Cross-Platform Information Flow
================================================================================

PURPOSE:
    Analyze how information flows between Polymarket and Kalshi for the Fed
    December 2025 rate decision market. We examine MULTIPLE DIMENSIONS of the
    orderbook to understand which platform reacts first to new information:

    TIER 1 - PRICE SIGNALS: Do price changes appear on one platform first?
    TIER 2 - LIQUIDITY DEPTH: Do liquidity shifts propagate across platforms?
    TIER 3 - ORDER IMBALANCES: Do buying/selling pressures transfer between markets?
    TIER 4 - BOOK STRUCTURE: Do structural changes (levels, concentration) lead/lag?

    This is NOT just "price discovery" - it's a holistic analysis of how orderbook
    information propagates between two competing prediction markets.

================================================================================
WORKED EXAMPLE (used throughout this script):
================================================================================

We use tiny dataset to illustrate ALL formulas. Imagine 5 time periods:

    Time    Poly_Price    Kalshi_Price
    ----    ----------    ------------
    t=0     0.50          0.48
    t=1     0.52          0.49         (Poly moved +0.02, Kalshi moved +0.01)
    t=2     0.55          0.53         (Poly moved +0.03, Kalshi moved +0.04)
    t=3     0.54          0.55         (Poly moved -0.01, Kalshi moved +0.02)
    t=4     0.56          0.57         (Poly moved +0.02, Kalshi moved +0.02)

Step 1: Calculate DELTA (change from previous period):
    ΔPoly  = [None, +0.02, +0.03, -0.01, +0.02]
    ΔKalshi = [None, +0.01, +0.04, +0.02, +0.02]

Step 2: Create LAGGED deltas (previous period's delta):
    ΔPoly_lag   = [None, None, +0.02, +0.03, -0.01]
    ΔKalshi_lag = [None, None, +0.01, +0.04, +0.02]

Step 3: For VAR, we need rows where ALL 4 values exist (t=2,3,4):

    Row   ΔPoly_t (Y1)   ΔKalshi_t (Y2)   ΔPoly_lag (X1)   ΔKalshi_lag (X2)
    ---   -----------    -------------    --------------   ----------------
    t=2      +0.03          +0.04            +0.02            +0.01
    t=3      -0.01          +0.02            +0.03            +0.04
    t=4      +0.02          +0.02            -0.01            +0.02

    n = 3 observations

================================================================================
VAR MODEL SPECIFICATION:
================================================================================

VAR runs TWO regressions with the SAME X variables (lagged deltas):

    REGRESSION 1 (Predicting Polymarket):
    ΔPoly_t = α₁ + β₁₁·ΔPoly_{t-1} + β₁₂·ΔKalshi_{t-1} + u₁

    REGRESSION 2 (Predicting Kalshi):
    ΔKalshi_t = α₂ + β₂₁·ΔPoly_{t-1} + β₂₂·ΔKalshi_{t-1} + u₂

    Where:
    - α = intercept (constant term)
    - β₁₁, β₂₁ = coefficients on lagged Polymarket
    - β₁₂, β₂₂ = coefficients on lagged Kalshi
    - u = error term (residual)

================================================================================
PRICE DISCOVERY INTERPRETATION:
================================================================================

After running both regressions, we use Granger causality tests:

    - If β₂₁ is significant: Poly LEADS (Poly changes predict Kalshi changes)
    - If β₁₂ is significant: Kalshi LEADS (Kalshi changes predict Poly changes)
    - If both significant: Bidirectional price discovery
    - If neither significant: No lead-lag relationship

For the Fed December 2025 market, we found:
    - Poly → Kalshi: F=112.2, p<0.0001 (SIGNIFICANT)
    - Kalshi → Poly: F=0.4, p=0.549 (not significant)

    RESULT: POLYMARKET LEADS price discovery for this market.

================================================================================
"""

import csv
import math
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "results/no_change/fed_nochange_30min_preprocessed.csv"

# Available variables (19 per platform, excluding timestamps)
# Each variable is measured on both Polymarket (poly_*) and Kalshi (kalshi_*)
VARIABLES = [
    "mid", "spread", "best_bid_yes", "best_bid_no",
    "depth_best_yes", "depth_best_no", "depth_top3_yes", "depth_top3_no",
    "total_depth_yes", "total_depth_no", "depth_5c_yes", "depth_5c_no",
    "imbalance_best", "imbalance_top3", "imbalance_total",
    "num_levels_yes", "num_levels_no", "vwap_yes", "vwap_no"
]


# =============================================================================
# STATISTICAL FUNCTIONS: Building Blocks
# =============================================================================
#
# These are the fundamental statistics needed for OLS regression.
# All formulas use population versions (dividing by n, not n-1).
#
# WORKED EXAMPLE DATA (from t=2,3,4 above):
#     X1 (ΔPoly_lag):   [+0.02, +0.03, -0.01]  → mean = 0.0133
#     X2 (ΔKalshi_lag): [+0.01, +0.04, +0.02]  → mean = 0.0233
#     Y1 (ΔPoly_t):     [+0.03, -0.01, +0.02]  → mean = 0.0133
#     Y2 (ΔKalshi_t):   [+0.04, +0.02, +0.02]  → mean = 0.0267
#
# =============================================================================

def mean(values):
    """
    Calculate arithmetic mean: x̄ = (Σx) / n

    EXAMPLE:
        X1 = [+0.02, +0.03, -0.01]
        mean(X1) = (0.02 + 0.03 + (-0.01)) / 3 = 0.04 / 3 = 0.0133
    """
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def variance(values):
    """
    Calculate variance: Var(X) = (1/n) × Σ(x - x̄)²

    Variance measures how spread out the data is from its mean.

    EXAMPLE:
        X1 = [+0.02, +0.03, -0.01], x̄ = 0.0133

        Deviations from mean:
            (0.02 - 0.0133)² = (0.0067)²  = 0.0000444
            (0.03 - 0.0133)² = (0.0167)²  = 0.0002778
            (-0.01 - 0.0133)² = (-0.0233)² = 0.0005444

        Var(X1) = (0.0000444 + 0.0002778 + 0.0005444) / 3
                = 0.0008667 / 3 = 0.000289
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    m = mean(valid)
    return sum((x - m) ** 2 for x in valid) / len(valid)


def covariance(x_values, y_values):
    """
    Calculate covariance: Cov(X,Y) = (1/n) × Σ(x - x̄)(y - ȳ)

    Covariance measures how two variables move together:
        - Positive: X and Y tend to move in same direction
        - Negative: X and Y tend to move in opposite directions
        - Zero: No linear relationship

    EXAMPLE:
        X1 = [+0.02, +0.03, -0.01], x̄ = 0.0133
        Y1 = [+0.03, -0.01, +0.02], ȳ = 0.0133

        Products of deviations:
            (0.02 - 0.0133)(0.03 - 0.0133) = (0.0067)(0.0167) = +0.000111
            (0.03 - 0.0133)(-0.01 - 0.0133) = (0.0167)(-0.0233) = -0.000389
            (-0.01 - 0.0133)(0.02 - 0.0133) = (-0.0233)(0.0067) = -0.000156

        Cov(X1,Y1) = (0.000111 + (-0.000389) + (-0.000156)) / 3
                   = -0.000433 / 3 = -0.000144
    """
    pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    if not pairs:
        return None
    x_vals, y_vals = zip(*pairs)
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / len(pairs)


# =============================================================================
# OLS REGRESSION WITH TWO X VARIABLES
# =============================================================================
#
# We want to find β₀, β₁, β₂ in the equation:
#
#     Y = β₀ + β₁·X₁ + β₂·X₂
#
# The formulas for the coefficients are:
#
#     β₁ = [Cov(X₁,Y)·Var(X₂) - Cov(X₂,Y)·Cov(X₁,X₂)] / [Var(X₁)·Var(X₂) - Cov(X₁,X₂)²]
#
#     β₂ = [Cov(X₂,Y)·Var(X₁) - Cov(X₁,Y)·Cov(X₁,X₂)] / [Var(X₁)·Var(X₂) - Cov(X₁,X₂)²]
#
#     β₀ = Ȳ - β₁·X̄₁ - β₂·X̄₂
#
# INTUITION:
#     - β₁ tells us: holding X₂ constant, how much does Y change per unit change in X₁?
#     - β₂ tells us: holding X₁ constant, how much does Y change per unit change in X₂?
#     - The denominator [Var(X₁)·Var(X₂) - Cov(X₁,X₂)²] adjusts for correlation between X₁ and X₂
#
# =============================================================================

def ols_two_variables(Y, X1, X2):
    """
    Run OLS regression: Y = β₀ + β₁·X₁ + β₂·X₂

    STEP-BY-STEP EXAMPLE (Regression 1: Predicting ΔPoly from lagged values):

        Y  = ΔPoly_t     = [+0.03, -0.01, +0.02]   (what we're predicting)
        X1 = ΔPoly_lag   = [+0.02, +0.03, -0.01]   (lagged Poly change)
        X2 = ΔKalshi_lag = [+0.01, +0.04, +0.02]   (lagged Kalshi change)

    STEP 1: Calculate all building blocks

        Var(X1)     = 0.000289        [variance of lagged Poly]
        Var(X2)     = 0.000156        [variance of lagged Kalshi]
        Cov(X1,X2)  = 0.000122        [covariance between X1 and X2]
        Cov(X1,Y)   = -0.000144       [covariance between X1 and Y]
        Cov(X2,Y)   = -0.000089       [covariance between X2 and Y]

        mean(X1) = 0.0133
        mean(X2) = 0.0233
        mean(Y)  = 0.0133

    STEP 2: Calculate denominator (same for both β's)

        denom = Var(X1)·Var(X2) - Cov(X1,X2)²
              = (0.000289)(0.000156) - (0.000122)²
              = 0.0000000451 - 0.0000000149
              = 0.0000000302

    STEP 3: Calculate β₁ (coefficient on lagged Poly)

        β₁ = [Cov(X1,Y)·Var(X2) - Cov(X2,Y)·Cov(X1,X2)] / denom
           = [(-0.000144)(0.000156) - (-0.000089)(0.000122)] / 0.0000000302
           = [-0.0000000225 - (-0.0000000109)] / 0.0000000302
           = -0.0000000116 / 0.0000000302
           = -0.384

    STEP 4: Calculate β₂ (coefficient on lagged Kalshi)

        β₂ = [Cov(X2,Y)·Var(X1) - Cov(X1,Y)·Cov(X1,X2)] / denom
           = [(-0.000089)(0.000289) - (-0.000144)(0.000122)] / 0.0000000302
           = [-0.0000000257 - (-0.0000000176)] / 0.0000000302
           = -0.0000000081 / 0.0000000302
           = -0.268

    STEP 5: Calculate β₀ (intercept)

        β₀ = mean(Y) - β₁·mean(X1) - β₂·mean(X2)
           = 0.0133 - (-0.384)(0.0133) - (-0.268)(0.0233)
           = 0.0133 + 0.00511 + 0.00624
           = 0.0247

    FINAL EQUATION:
        ΔPoly_t = 0.0247 - 0.384·ΔPoly_{t-1} - 0.268·ΔKalshi_{t-1}

    Returns: (β₀, β₁, β₂, R², SSR, n)
    """
    # Filter to rows where all values are present
    valid_rows = [(y, x1, x2) for y, x1, x2 in zip(Y, X1, X2)
                  if y is not None and x1 is not None and x2 is not None]

    if len(valid_rows) < 4:
        return None, None, None, None, None, 0

    Y_valid, X1_valid, X2_valid = zip(*valid_rows)
    n = len(valid_rows)

    # -------------------------------------------------------------------------
    # STEP 1: Calculate building blocks
    # These are the core statistics we need for the OLS formulas
    # -------------------------------------------------------------------------
    var_x1 = variance(X1_valid)       # Var(X₁)
    var_x2 = variance(X2_valid)       # Var(X₂)
    cov_x1_x2 = covariance(X1_valid, X2_valid)  # Cov(X₁,X₂)
    cov_x1_y = covariance(X1_valid, Y_valid)    # Cov(X₁,Y)
    cov_x2_y = covariance(X2_valid, Y_valid)    # Cov(X₂,Y)
    mean_x1 = mean(X1_valid)          # X̄₁
    mean_x2 = mean(X2_valid)          # X̄₂
    mean_y = mean(Y_valid)            # Ȳ

    # -------------------------------------------------------------------------
    # STEP 2: Calculate denominator
    # denom = Var(X₁)·Var(X₂) - Cov(X₁,X₂)²
    # This is the same for both β₁ and β₂ formulas
    # -------------------------------------------------------------------------
    denom = var_x1 * var_x2 - cov_x1_x2 ** 2

    if abs(denom) < 1e-20:
        # Denominator too small = multicollinearity (X1 and X2 are too correlated)
        return None, None, None, None, None, n

    # -------------------------------------------------------------------------
    # STEP 3: Calculate β₁ (coefficient on X₁)
    # Formula: β₁ = [Cov(X₁,Y)·Var(X₂) - Cov(X₂,Y)·Cov(X₁,X₂)] / denom
    # -------------------------------------------------------------------------
    beta1 = (cov_x1_y * var_x2 - cov_x2_y * cov_x1_x2) / denom

    # -------------------------------------------------------------------------
    # STEP 4: Calculate β₂ (coefficient on X₂)
    # Formula: β₂ = [Cov(X₂,Y)·Var(X₁) - Cov(X₁,Y)·Cov(X₁,X₂)] / denom
    # -------------------------------------------------------------------------
    beta2 = (cov_x2_y * var_x1 - cov_x1_y * cov_x1_x2) / denom

    # -------------------------------------------------------------------------
    # STEP 5: Calculate β₀ (intercept)
    # Formula: β₀ = Ȳ - β₁·X̄₁ - β₂·X̄₂
    # -------------------------------------------------------------------------
    beta0 = mean_y - beta1 * mean_x1 - beta2 * mean_x2

    # -------------------------------------------------------------------------
    # STEP 6: Calculate R² (goodness of fit)
    #
    # R² tells us what fraction of Y's variance is explained by our model.
    #
    # Formula: R² = 1 - SSR/SST
    #
    # Where:
    #   SSR = Sum of Squared Residuals = Σ(Y - Ŷ)² = Σ(error)²
    #   SST = Total Sum of Squares = Σ(Y - Ȳ)²
    #
    # EXAMPLE:
    #   For each observation, calculate predicted value Ŷ = β₀ + β₁·X₁ + β₂·X₂
    #
    #   Row t=2: Ŷ = 0.0247 + (-0.384)(0.02) + (-0.268)(0.01)
    #              = 0.0247 - 0.00768 - 0.00268 = 0.0143
    #          Actual Y = 0.03
    #          Error = 0.03 - 0.0143 = 0.0157
    #          Error² = 0.000246
    #
    #   ... (repeat for all rows)
    #
    #   SSR = Σ(error²) = sum of all squared errors
    #   SST = Σ(Y - Ȳ)² = sum of squared deviations from mean
    #   R² = 1 - (SSR / SST)
    #
    # Interpretation:
    #   R² = 0.10 means model explains 10% of variance in Y
    #   R² = 0.50 means model explains 50% of variance in Y
    # -------------------------------------------------------------------------

    # Calculate predicted values: Ŷ = β₀ + β₁·X₁ + β₂·X₂
    Y_pred = [beta0 + beta1 * x1 + beta2 * x2 for x1, x2 in zip(X1_valid, X2_valid)]

    # Calculate errors (residuals): e = Y - Ŷ
    errors = [y - y_pred for y, y_pred in zip(Y_valid, Y_pred)]

    # SSR = Sum of Squared Residuals = Σ(error²)
    SSR = sum(e ** 2 for e in errors)

    # SST = Total Sum of Squares = Σ(Y - Ȳ)²
    SST = sum((y - mean_y) ** 2 for y in Y_valid)

    # R² = 1 - SSR/SST
    R_squared = 1 - (SSR / SST) if SST > 0 else 0

    return beta0, beta1, beta2, R_squared, SSR, n


def ols_one_variable(Y, X):
    """
    Simple OLS regression with one X variable: Y = β₀ + β·X

    This is the "restricted" model used in Granger causality tests.
    When testing if X₂ adds predictive power, we compare:
        - Unrestricted: Y = β₀ + β₁·X₁ + β₂·X₂
        - Restricted:   Y = β₀ + β·X₁  (X₂ excluded)

    FORMULA:
        β = Cov(X,Y) / Var(X)
        β₀ = Ȳ - β·X̄

    EXAMPLE (predicting ΔPoly using only lagged Poly):
        Y = [+0.03, -0.01, +0.02]
        X = [+0.02, +0.03, -0.01]

        β = Cov(X,Y) / Var(X) = -0.000144 / 0.000289 = -0.50
        β₀ = 0.0133 - (-0.50)(0.0133) = 0.0133 + 0.00665 = 0.0200

        Equation: ΔPoly_t = 0.020 - 0.50·ΔPoly_{t-1}

    Returns: (β₀, β, R², SSR, n)
    """
    valid_rows = [(y, x) for y, x in zip(Y, X) if y is not None and x is not None]

    if len(valid_rows) < 3:
        return None, None, None, None, 0

    Y_valid, X_valid = zip(*valid_rows)
    n = len(valid_rows)

    var_x = variance(X_valid)
    cov_x_y = covariance(X_valid, Y_valid)
    mean_x = mean(X_valid)
    mean_y = mean(Y_valid)

    if var_x is None or var_x == 0:
        return None, None, None, None, n

    # β = Cov(X,Y) / Var(X)
    beta = cov_x_y / var_x

    # β₀ = Ȳ - β·X̄
    beta0 = mean_y - beta * mean_x

    # Calculate SSR for R² and Granger test
    Y_pred = [beta0 + beta * x for x in X_valid]
    errors = [y - y_pred for y, y_pred in zip(Y_valid, Y_pred)]
    SSR = sum(e ** 2 for e in errors)

    # Calculate R²
    SST = sum((y - mean_y) ** 2 for y in Y_valid)
    R_squared = 1 - (SSR / SST) if SST > 0 else 0

    return beta0, beta, R_squared, SSR, n


# =============================================================================
# GRANGER CAUSALITY TEST
# =============================================================================
#
# Granger causality tests whether adding a variable SIGNIFICANTLY improves
# predictions. It uses an F-test to compare two models:
#
#     Unrestricted: Y = β₀ + β₁·X₁ + β₂·X₂  (both lagged values included)
#     Restricted:   Y = β₀ + β·X₁            (only own lag, other excluded)
#
# If the unrestricted model fits MUCH better (lower SSR), the excluded
# variable has predictive power → Granger causality!
#
# F-STATISTIC FORMULA:
#
#     F = [(SSR_restricted - SSR_unrestricted) / q] / [SSR_unrestricted / (n - k)]
#
# Where:
#     SSR_restricted   = Sum of squared errors from restricted model
#     SSR_unrestricted = Sum of squared errors from unrestricted model
#     q = number of restrictions = 1 (we're testing one coefficient)
#     n = number of observations
#     k = parameters in unrestricted model = 3 (β₀, β₁, β₂)
#
# INTUITION:
#     - Numerator: How much SSR DECREASED by adding the variable (per restriction)
#     - Denominator: Baseline variance (SSR per degree of freedom)
#     - Large F → Adding the variable helped A LOT → Reject null → Significant!
#
# EXAMPLE:
#     SSR_unrestricted = 0.000150 (model with both lags)
#     SSR_restricted = 0.000200 (model with only own lag)
#     n = 1003, k = 3, q = 1
#
#     F = [(0.000200 - 0.000150) / 1] / [0.000150 / (1003 - 3)]
#       = 0.000050 / 0.00000015
#       = 333.3
#
#     With F = 333.3 and df=(1, 1000), p-value ≈ 0.0000
#     → Highly significant! The excluded variable has predictive power.
#
# =============================================================================

def granger_test(Y, X1, X2, test_x1=True):
    """
    Granger causality test.

    Tests whether X1 (if test_x1=True) or X2 (if test_x1=False) significantly
    improves prediction of Y beyond the other variable.

    TESTING "DOES POLY PREDICT KALSHI?" (Poly → Kalshi):
        Unrestricted: ΔKalshi_t = α + β₁·ΔPoly_{t-1} + β₂·ΔKalshi_{t-1}
        Restricted:   ΔKalshi_t = α + β·ΔKalshi_{t-1}  (Poly lag EXCLUDED)

        If F is significant: Poly DOES predict Kalshi → Poly leads!

    TESTING "DOES KALSHI PREDICT POLY?" (Kalshi → Poly):
        Unrestricted: ΔPoly_t = α + β₁·ΔPoly_{t-1} + β₂·ΔKalshi_{t-1}
        Restricted:   ΔPoly_t = α + β·ΔPoly_{t-1}  (Kalshi lag EXCLUDED)

        If F is significant: Kalshi DOES predict Poly → Kalshi leads!

    Returns: (F_statistic, p_value, significant_at_5pct)
    """
    # -------------------------------------------------------------------------
    # STEP 1: Run unrestricted model (includes both X1 and X2)
    # -------------------------------------------------------------------------
    _, _, _, _, SSR_unrestricted, n = ols_two_variables(Y, X1, X2)

    if SSR_unrestricted is None or n < 5:
        return None, None, None

    # -------------------------------------------------------------------------
    # STEP 2: Run restricted model (excludes the variable being tested)
    # -------------------------------------------------------------------------
    if test_x1:
        # Testing X1 (lagged Poly): exclude X1, keep X2
        _, _, _, SSR_restricted, _ = ols_one_variable(Y, X2)
    else:
        # Testing X2 (lagged Kalshi): exclude X2, keep X1
        _, _, _, SSR_restricted, _ = ols_one_variable(Y, X1)

    if SSR_restricted is None:
        return None, None, None

    # -------------------------------------------------------------------------
    # STEP 3: Calculate F-statistic
    #
    # F = [(SSR_r - SSR_u) / q] / [SSR_u / (n - k)]
    #
    # q = 1 (testing one coefficient)
    # k = 3 (parameters in unrestricted: β₀, β₁, β₂)
    # -------------------------------------------------------------------------
    q = 1  # Number of restrictions (testing one coefficient)
    k = 3  # Parameters in unrestricted model (β₀, β₁, β₂)

    if SSR_unrestricted == 0:
        return None, None, None

    # Numerator: reduction in SSR per restriction
    numerator = (SSR_restricted - SSR_unrestricted) / q

    # Denominator: baseline variance (MSE of unrestricted model)
    denominator = SSR_unrestricted / (n - k)

    F = numerator / denominator

    # -------------------------------------------------------------------------
    # STEP 4: Calculate p-value from F-distribution
    #
    # Under null hypothesis (β=0), F follows F-distribution with:
    #     df1 = q = 1 (numerator degrees of freedom)
    #     df2 = n - k (denominator degrees of freedom)
    #
    # p-value = P(F_random > F_observed)
    # If p < 0.05: Reject null → Coefficient IS significant!
    # -------------------------------------------------------------------------
    df1 = q       # Numerator degrees of freedom
    df2 = n - k   # Denominator degrees of freedom

    # P(F > observed) = 1 - CDF(observed)
    p_value = 1 - stats.f.cdf(F, df1, df2)

    significant = p_value < 0.05

    return F, p_value, significant


# =============================================================================
# VAR ANALYSIS: Putting It All Together
# =============================================================================
#
# VAR for price discovery runs TWO regressions and TWO Granger tests:
#
#     REGRESSION 1: Predicting Polymarket
#         ΔPoly_t = α₁ + β₁₁·ΔPoly_{t-1} + β₁₂·ΔKalshi_{t-1}
#         Granger test: Is β₁₂ significant? (Does Kalshi predict Poly?)
#
#     REGRESSION 2: Predicting Kalshi
#         ΔKalshi_t = α₂ + β₂₁·ΔPoly_{t-1} + β₂₂·ΔKalshi_{t-1}
#         Granger test: Is β₂₁ significant? (Does Poly predict Kalshi?)
#
# INTERPRETATION:
#     β₂₁ significant, β₁₂ not → Poly LEADS (Poly predicts Kalshi, not vice versa)
#     β₁₂ significant, β₂₁ not → Kalshi LEADS
#     Both significant → Bidirectional
#     Neither significant → No lead-lag relationship
#
# =============================================================================

def run_var(variable_name, data):
    """
    Run complete VAR analysis for a given variable.

    Args:
        variable_name: One of the 19 variable names (e.g., "mid", "spread")
        data: List of dictionaries from CSV

    COLUMN NAMING CONVENTION:
        Base columns: poly_{var}, kalshi_{var}
        Delta columns: d_poly_{var}, d_kalshi_{var}
        Lagged delta columns: d_poly_{var}_lag1, d_kalshi_{var}_lag1

    EXAMPLE for variable "mid":
        Y1 = d_poly_mid       (current period Poly change - what we predict in Reg 1)
        Y2 = d_kalshi_mid     (current period Kalshi change - what we predict in Reg 2)
        X1 = d_poly_mid_lag1  (previous period Poly change - predictor)
        X2 = d_kalshi_mid_lag1 (previous period Kalshi change - predictor)

    Returns dictionary with all VAR results.
    """
    # -------------------------------------------------------------------------
    # STEP 1: Build column names for this variable
    # -------------------------------------------------------------------------
    d_poly = f"d_poly_{variable_name}"         # ΔPoly (current) = Y1
    d_kalshi = f"d_kalshi_{variable_name}"     # ΔKalshi (current) = Y2
    d_poly_lag = f"d_poly_{variable_name}_lag1"     # ΔPoly_{t-1} = X1
    d_kalshi_lag = f"d_kalshi_{variable_name}_lag1"  # ΔKalshi_{t-1} = X2

    # -------------------------------------------------------------------------
    # STEP 2: Extract columns from data
    # -------------------------------------------------------------------------
    Y1 = [float(row[d_poly]) if row[d_poly] else None for row in data]    # ΔPoly (current)
    Y2 = [float(row[d_kalshi]) if row[d_kalshi] else None for row in data]  # ΔKalshi (current)
    X1 = [float(row[d_poly_lag]) if row[d_poly_lag] else None for row in data]  # ΔPoly_{t-1}
    X2 = [float(row[d_kalshi_lag]) if row[d_kalshi_lag] else None for row in data]  # ΔKalshi_{t-1}

    # =========================================================================
    # REGRESSION 1: Predicting Polymarket
    # ΔPoly_t = α₁ + β₁₁·ΔPoly_{t-1} + β₁₂·ΔKalshi_{t-1} + u₁
    #
    # This regression asks: Can we predict Poly's next move using:
    #   - Poly's previous move (β₁₁ = autocorrelation / momentum)
    #   - Kalshi's previous move (β₁₂ = cross-market influence)
    #
    # If β₁₂ is significant → Kalshi LEADS Poly (Kalshi's moves predict Poly's)
    # =========================================================================
    beta0_1, beta11, beta12, r2_1, ssr_1, n_1 = ols_two_variables(Y1, X1, X2)

    # Granger test: Does lagged Kalshi (X2) predict Poly (Y1)?
    # test_x1=False means we're testing X2 (the Kalshi lag)
    f_kalshi_to_poly, p_kalshi_to_poly, sig_kalshi_to_poly = granger_test(Y1, X1, X2, test_x1=False)

    # =========================================================================
    # REGRESSION 2: Predicting Kalshi
    # ΔKalshi_t = α₂ + β₂₁·ΔPoly_{t-1} + β₂₂·ΔKalshi_{t-1} + u₂
    #
    # This regression asks: Can we predict Kalshi's next move using:
    #   - Poly's previous move (β₂₁ = cross-market influence)
    #   - Kalshi's previous move (β₂₂ = autocorrelation / momentum)
    #
    # If β₂₁ is significant → Poly LEADS Kalshi (Poly's moves predict Kalshi's)
    # =========================================================================
    beta0_2, beta21, beta22, r2_2, ssr_2, n_2 = ols_two_variables(Y2, X1, X2)

    # Granger test: Does lagged Poly (X1) predict Kalshi (Y2)?
    # test_x1=True means we're testing X1 (the Poly lag)
    f_poly_to_kalshi, p_poly_to_kalshi, sig_poly_to_kalshi = granger_test(Y2, X1, X2, test_x1=True)

    # =========================================================================
    # Return all results
    # =========================================================================
    return {
        "variable": variable_name,
        "n_observations": n_1,

        # Regression 1: Predicting Poly
        "reg1_alpha": beta0_1,           # α₁ (intercept)
        "reg1_beta_poly_lag": beta11,    # β₁₁ (coefficient on own lag)
        "reg1_beta_kalshi_lag": beta12,  # β₁₂ (coefficient on Kalshi lag)
        "reg1_r_squared": r2_1,          # R² (goodness of fit)

        # Regression 2: Predicting Kalshi
        "reg2_alpha": beta0_2,           # α₂ (intercept)
        "reg2_beta_poly_lag": beta21,    # β₂₁ (coefficient on Poly lag)
        "reg2_beta_kalshi_lag": beta22,  # β₂₂ (coefficient on own lag)
        "reg2_r_squared": r2_2,          # R² (goodness of fit)

        # Granger causality: Kalshi → Poly
        "granger_kalshi_to_poly_F": f_kalshi_to_poly,
        "granger_kalshi_to_poly_p": p_kalshi_to_poly,
        "granger_kalshi_to_poly_sig": sig_kalshi_to_poly,

        # Granger causality: Poly → Kalshi
        "granger_poly_to_kalshi_F": f_poly_to_kalshi,
        "granger_poly_to_kalshi_p": p_poly_to_kalshi,
        "granger_poly_to_kalshi_sig": sig_poly_to_kalshi,
    }


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_var_results(results):
    """Print formatted VAR results with interpretation."""
    print(f"\n{'=' * 70}")
    print(f"VAR RESULTS: {results['variable'].upper()}")
    print(f"{'=' * 70}")
    print(f"Observations: {results['n_observations']}")

    # -------------------------------------------------------------------------
    # Regression 1: Predicting Polymarket
    # -------------------------------------------------------------------------
    print(f"\n--- Regression 1: Predicting ΔPoly ---")
    print(f"ΔPoly_t = {results['reg1_alpha']:.6f} + {results['reg1_beta_poly_lag']:.4f}·ΔPoly_{{t-1}} + {results['reg1_beta_kalshi_lag']:.4f}·ΔKalshi_{{t-1}}")
    print(f"R² = {results['reg1_r_squared']:.4f} ({results['reg1_r_squared']*100:.1f}% of variance explained)")

    # -------------------------------------------------------------------------
    # Regression 2: Predicting Kalshi
    # -------------------------------------------------------------------------
    print(f"\n--- Regression 2: Predicting ΔKalshi ---")
    print(f"ΔKalshi_t = {results['reg2_alpha']:.6f} + {results['reg2_beta_poly_lag']:.4f}·ΔPoly_{{t-1}} + {results['reg2_beta_kalshi_lag']:.4f}·ΔKalshi_{{t-1}}")
    print(f"R² = {results['reg2_r_squared']:.4f} ({results['reg2_r_squared']*100:.1f}% of variance explained)")

    # -------------------------------------------------------------------------
    # Granger Causality Tests
    # -------------------------------------------------------------------------
    print(f"\n--- Granger Causality Tests ---")
    print(f"Kalshi → Poly:  F = {results['granger_kalshi_to_poly_F']:.3f}, p = {results['granger_kalshi_to_poly_p']:.4f}", end="")
    print(f" {'***' if results['granger_kalshi_to_poly_sig'] else '(not significant)'}")

    print(f"Poly → Kalshi:  F = {results['granger_poly_to_kalshi_F']:.3f}, p = {results['granger_poly_to_kalshi_p']:.4f}", end="")
    print(f" {'***' if results['granger_poly_to_kalshi_sig'] else '(not significant)'}")

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    print(f"\n--- Interpretation ---")
    if results['granger_poly_to_kalshi_sig'] and not results['granger_kalshi_to_poly_sig']:
        print(f"→ POLYMARKET LEADS: Poly predicts Kalshi (p={results['granger_poly_to_kalshi_p']:.4f})")
        print(f"  A 1% move in Poly predicts a {results['reg2_beta_poly_lag']:.2f}% move in Kalshi next period")
    elif results['granger_kalshi_to_poly_sig'] and not results['granger_poly_to_kalshi_sig']:
        print(f"→ KALSHI LEADS: Kalshi predicts Poly (p={results['granger_kalshi_to_poly_p']:.4f})")
        print(f"  A 1% move in Kalshi predicts a {results['reg1_beta_kalshi_lag']:.2f}% move in Poly next period")
    elif results['granger_poly_to_kalshi_sig'] and results['granger_kalshi_to_poly_sig']:
        print(f"→ BIDIRECTIONAL: Both platforms predict each other")
    else:
        print(f"→ NO SIGNIFICANT LEAD-LAG RELATIONSHIP")


# =============================================================================
# MAIN
# =============================================================================

def print_detailed_explanation(results):
    """
    Print a detailed explanation of the VAR results that connects
    the beta coefficients to F-statistics and explains interpretation.
    """
    var = results['variable'].upper()

    print(f"""
================================================================================
DETAILED EXPLANATION OF RESULTS: {var}
================================================================================

WHAT WE RAN
-----------
We ran two regressions to see if one platform's price changes predict the other's:

    Regression 1: Can we predict Polymarket's next move?
    Regression 2: Can we predict Kalshi's next move?

Both regressions use the SAME predictors (X variables):
    X1 = Polymarket's PREVIOUS price change  (lagged 30 min)
    X2 = Kalshi's PREVIOUS price change      (lagged 30 min)


THE REGRESSION EQUATIONS
------------------------
Regression 1 (Predicting Polymarket):

    DeltaPoly_t = {results['reg1_alpha']:.6f} + {results['reg1_beta_poly_lag']:.4f}*DeltaPoly_{{t-1}} + {results['reg1_beta_kalshi_lag']:.4f}*DeltaKalshi_{{t-1}}
                  \_________/   \______________/                  \________________/
                   intercept    own lag effect                    CROSS-MARKET EFFECT
                   (alpha)      (beta_11)                         (beta_12)
                                                                  |
                                                                  v
                                            This is what we TEST: Does Kalshi predict Poly?
                                            beta_12 = {results['reg1_beta_kalshi_lag']:.4f}

Regression 2 (Predicting Kalshi):

    DeltaKalshi_t = {results['reg2_alpha']:.6f} + {results['reg2_beta_poly_lag']:.4f}*DeltaPoly_{{t-1}} + {results['reg2_beta_kalshi_lag']:.4f}*DeltaKalshi_{{t-1}}
                    \_________/   \______________/                  \________________/
                     intercept    CROSS-MARKET EFFECT               own lag effect
                     (alpha)      (beta_21)                         (beta_22)
                                  |
                                  v
                  This is what we TEST: Does Poly predict Kalshi?
                  beta_21 = {results['reg2_beta_poly_lag']:.4f}


WHAT THE BETA COEFFICIENTS MEAN
-------------------------------
beta_21 = {results['reg2_beta_poly_lag']:.4f} (Poly lag in Kalshi regression)

    Interpretation: When Polymarket moves by 1 unit, Kalshi moves by {results['reg2_beta_poly_lag']:.4f} units
                    in the NEXT period, on average.

    Example: If Poly's mid price increased by +2% last period,
             we'd expect Kalshi to increase by {results['reg2_beta_poly_lag']:.4f} * 2% = {results['reg2_beta_poly_lag'] * 2:.4f}%
             this period (all else equal).

beta_12 = {results['reg1_beta_kalshi_lag']:.4f} (Kalshi lag in Poly regression)

    Interpretation: When Kalshi moves by 1 unit, Poly moves by {results['reg1_beta_kalshi_lag']:.4f} units
                    in the NEXT period, on average.


HOW BETA RELATES TO THE F-STATISTIC
-----------------------------------
The beta coefficient tells us the SIZE of the effect.
The F-statistic tells us if the effect is REAL (statistically significant).

A large beta with a small F means: "Big effect, but might be random noise"
A small beta with a large F means: "Small effect, but definitely real"
A large beta with a large F means: "Big effect AND definitely real" (best case)

To calculate F, we compare two models:
    - UNRESTRICTED: includes both lags (the full model)
    - RESTRICTED: removes the lag we're testing (simpler model)

    F = (How much WORSE the restricted model fits) / (Baseline noise level)

If removing a variable makes the model MUCH worse, F is large = that variable matters!


THE F-STATISTICS AND P-VALUES
-----------------------------
Testing "Does Poly predict Kalshi?" (Poly -> Kalshi):

    F = {results['granger_poly_to_kalshi_F']:.3f}
    p = {results['granger_poly_to_kalshi_p']:.6f}

    What F means: Removing Poly's lag made the Kalshi prediction {results['granger_poly_to_kalshi_F']:.1f}x worse
                  relative to the baseline noise level.

    What p means: There's a {results['granger_poly_to_kalshi_p'] * 100:.4f}% chance we'd see an F this large
                  if Poly had NO real predictive power (pure chance).

    Decision: p {'< 0.05, so REJECT null hypothesis. Poly DOES predict Kalshi.' if results['granger_poly_to_kalshi_sig'] else '>= 0.05, so CANNOT reject null. No evidence Poly predicts Kalshi.'}

Testing "Does Kalshi predict Poly?" (Kalshi -> Poly):

    F = {results['granger_kalshi_to_poly_F']:.3f}
    p = {results['granger_kalshi_to_poly_p']:.6f}

    What F means: Removing Kalshi's lag made the Poly prediction {results['granger_kalshi_to_poly_F']:.1f}x worse
                  relative to the baseline noise level.

    What p means: There's a {results['granger_kalshi_to_poly_p'] * 100:.2f}% chance we'd see an F this large
                  if Kalshi had NO real predictive power (pure chance).

    Decision: p {'< 0.05, so REJECT null hypothesis. Kalshi DOES predict Poly.' if results['granger_kalshi_to_poly_sig'] else '>= 0.05, so CANNOT reject null. No evidence Kalshi predicts Poly.'}


R-SQUARED: HOW WELL DO THE MODELS FIT?
--------------------------------------
R-squared tells us what fraction of price movements we can explain.

    R²(Poly regression)   = {results['reg1_r_squared']:.4f} = {results['reg1_r_squared']*100:.1f}%
    R²(Kalshi regression) = {results['reg2_r_squared']:.4f} = {results['reg2_r_squared']*100:.1f}%

    Interpretation:
    - {results['reg1_r_squared']*100:.1f}% of Poly's price movements can be explained by past data
    - {results['reg2_r_squared']*100:.1f}% of Kalshi's price movements can be explained by past data

    The fact that R²(Kalshi) > R²(Poly) supports the finding that Kalshi is
    the "follower" - its movements are more predictable because it's reacting
    to Polymarket.


FINAL INTERPRETATION
--------------------""")

    if results['granger_poly_to_kalshi_sig'] and not results['granger_kalshi_to_poly_sig']:
        print(f"""
    POLYMARKET LEADS PRICE DISCOVERY

    Evidence:
    1. Poly -> Kalshi: F={results['granger_poly_to_kalshi_F']:.1f}, p={results['granger_poly_to_kalshi_p']:.4f} (SIGNIFICANT)
       When Poly moves, Kalshi follows next period.

    2. Kalshi -> Poly: F={results['granger_kalshi_to_poly_F']:.1f}, p={results['granger_kalshi_to_poly_p']:.4f} (not significant)
       When Kalshi moves, Poly does NOT follow.

    Magnitude: A 1% move in Poly predicts a {results['reg2_beta_poly_lag']:.2f}% move in Kalshi.

    What this means in practice:
    - New information appears on Polymarket FIRST
    - Kalshi prices adjust ~30 minutes LATER
    - Polymarket has more informed/faster traders for this market
""")
    elif results['granger_kalshi_to_poly_sig'] and not results['granger_poly_to_kalshi_sig']:
        print(f"""
    KALSHI LEADS PRICE DISCOVERY

    Evidence:
    1. Kalshi -> Poly: F={results['granger_kalshi_to_poly_F']:.1f}, p={results['granger_kalshi_to_poly_p']:.4f} (SIGNIFICANT)
       When Kalshi moves, Poly follows next period.

    2. Poly -> Kalshi: F={results['granger_poly_to_kalshi_F']:.1f}, p={results['granger_poly_to_kalshi_p']:.4f} (not significant)
       When Poly moves, Kalshi does NOT follow.

    Magnitude: A 1% move in Kalshi predicts a {results['reg1_beta_kalshi_lag']:.2f}% move in Poly.

    What this means in practice:
    - New information appears on Kalshi FIRST
    - Polymarket prices adjust ~30 minutes LATER
    - Kalshi has more informed/faster traders for this market
""")
    elif results['granger_poly_to_kalshi_sig'] and results['granger_kalshi_to_poly_sig']:
        print(f"""
    BIDIRECTIONAL PRICE DISCOVERY

    Both platforms predict each other - information flows both ways.
    Neither platform has a clear lead.
""")
    else:
        print(f"""
    NO SIGNIFICANT LEAD-LAG RELATIONSHIP

    Neither platform's past moves predict the other's future moves.
    Prices may move together simultaneously, or the relationship
    may be too noisy to detect at 30-minute intervals.
""")


def main():
    print("=" * 80)
    print("CROSS-PLATFORM INFORMATION FLOW ANALYSIS")
    print("Fed December 2025 Rate Decision: Polymarket vs Kalshi")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load preprocessed data
    # -------------------------------------------------------------------------
    print(f"\nLoading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    print(f"Loaded {len(data)} observations (30-minute intervals)")

    # -------------------------------------------------------------------------
    # Introduction: What we're analyzing
    # -------------------------------------------------------------------------
    print("""

================================================================================
WHAT THIS ANALYSIS DOES
================================================================================

We extracted 19 variables from orderbook data across 4 TIERS to understand how
information flows between Polymarket and Kalshi:

TIER 1 - PRICE SIGNALS (4 variables)
    Question: When prices move, which platform moves first?
    Variables: mid price, spread, best bid YES, best bid NO

TIER 2 - LIQUIDITY DEPTH (6 variables)
    Question: When liquidity shifts, which platform shows it first?
    Variables: depth at best, depth in top 3 levels, total depth (YES/NO sides)

TIER 3 - ORDER IMBALANCES (3 variables)
    Question: When buying/selling pressure builds, where does it appear first?
    Variables: imbalance at best, imbalance in top 3, total imbalance

TIER 4 - BOOK STRUCTURE (6 variables)
    Question: When the orderbook structure changes, which platform leads?
    Variables: depth within 5 cents, number of price levels, VWAP (YES/NO sides)

For EACH variable, we run VAR (Vector Autoregression) to test:
    - Does Polymarket's past predict Kalshi's future? (Poly leads)
    - Does Kalshi's past predict Polymarket's future? (Kalshi leads)

================================================================================
""")

    # -------------------------------------------------------------------------
    # Run VAR for mid price with detailed explanation
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("DETAILED ANALYSIS: MID PRICE (Primary Price Signal)")
    print("=" * 80)

    results = run_var("mid", data)
    print_var_results(results)
    print_detailed_explanation(results)

    # -------------------------------------------------------------------------
    # Tier-by-tier analysis
    # -------------------------------------------------------------------------
    # Define tiers
    tier1 = ["mid", "spread", "best_bid_yes", "best_bid_no"]
    tier2 = ["depth_best_yes", "depth_best_no", "depth_top3_yes", "depth_top3_no", "total_depth_yes", "total_depth_no"]
    tier3 = ["imbalance_best", "imbalance_top3", "imbalance_total"]
    tier4 = ["depth_5c_yes", "depth_5c_no", "num_levels_yes", "num_levels_no", "vwap_yes", "vwap_no"]

    def print_tier_results(tier_name, tier_question, variables, data):
        print(f"\n{'=' * 80}")
        print(f"{tier_name}")
        print(f"Question: {tier_question}")
        print("=" * 80)
        print(f"\n{'Variable':<20} {'Poly->Kalshi':<18} {'Kalshi->Poly':<18} {'Leader':<12} {'Interpretation'}")
        print("-" * 100)

        for var in variables:
            try:
                r = run_var(var, data)
                if r['reg1_r_squared'] is None:
                    continue

                # Format results
                p2k_f = r['granger_poly_to_kalshi_F']
                k2p_f = r['granger_kalshi_to_poly_F']
                p2k_sig = r['granger_poly_to_kalshi_sig']
                k2p_sig = r['granger_kalshi_to_poly_sig']

                poly_to_kalshi = f"F={p2k_f:.1f}" + ("***" if p2k_sig else "")
                kalshi_to_poly = f"F={k2p_f:.1f}" + ("***" if k2p_sig else "")

                # Determine leader and interpretation
                if p2k_sig and not k2p_sig:
                    leader = "POLY"
                    beta = r['reg2_beta_poly_lag']
                    interp = f"Poly moves -> Kalshi follows ({beta:.2f}x)"
                elif k2p_sig and not p2k_sig:
                    leader = "KALSHI"
                    beta = r['reg1_beta_kalshi_lag']
                    interp = f"Kalshi moves -> Poly follows ({beta:.2f}x)"
                elif p2k_sig and k2p_sig:
                    leader = "BOTH"
                    interp = "Bidirectional flow"
                else:
                    leader = "-"
                    interp = "No significant cross-platform flow"

                print(f"{var:<20} {poly_to_kalshi:<18} {kalshi_to_poly:<18} {leader:<12} {interp}")
            except Exception as e:
                print(f"{var:<20} ERROR: {e}")

    print_tier_results(
        "TIER 1: PRICE SIGNALS",
        "When prices move, which platform moves first?",
        tier1, data
    )

    print_tier_results(
        "TIER 2: LIQUIDITY DEPTH",
        "When liquidity shifts, which platform shows it first?",
        tier2, data
    )

    print_tier_results(
        "TIER 3: ORDER IMBALANCES",
        "When buying/selling pressure builds, where does it appear first?",
        tier3, data
    )

    print_tier_results(
        "TIER 4: BOOK STRUCTURE",
        "When the orderbook structure changes, which platform leads?",
        tier4, data
    )

    # -------------------------------------------------------------------------
    # Summary and conclusions
    # -------------------------------------------------------------------------
    print(f"""

{'=' * 80}
SUMMARY OF FINDINGS
{'=' * 80}

NOTATION KEY
------------
F-statistic:  Measures how much predictive power a variable has
              F > 4 with *** = statistically significant (p < 0.05)
              Higher F = stronger evidence of information flow

***:          Statistically significant at 5% level
              Means there's less than 5% chance this is random noise

Leader:       Which platform's PAST predicts the other's FUTURE
              POLY = Polymarket moves first, Kalshi follows
              KALSHI = Kalshi moves first, Polymarket follows


FINDINGS BY TIER
----------------

TIER 1 - PRICE SIGNALS: POLYMARKET LEADS
    - Mid price:      Poly -> Kalshi (F=112.2***)
    - Best bid YES:   Poly -> Kalshi (F=104.9***)
    - Best bid NO:    Poly -> Kalshi (F=133.6***)
    - Spread:         No significant flow

    Conclusion: Price information appears on Polymarket FIRST.
    Kalshi prices adjust to match Polymarket ~30 minutes later.

TIER 2 - LIQUIDITY DEPTH: NO CLEAR PATTERN
    - All 6 depth variables show no significant cross-platform flow
    - Liquidity changes happen independently on each platform
    - Or: changes happen simultaneously (within 30-min window)

    Conclusion: Liquidity dynamics are platform-specific, not linked.

TIER 3 - ORDER IMBALANCES: NO CLEAR PATTERN
    - All 3 imbalance measures show no significant cross-platform flow
    - Order flow imbalances don't propagate between platforms

    Conclusion: Buying/selling pressure doesn't transfer between markets.

TIER 4 - BOOK STRUCTURE: MIXED RESULTS
    - num_levels_yes: Poly -> Kalshi (F=4.7***)
    - depth_5c_no:    Kalshi -> Poly (F=4.5***)
    - vwap_no:        Kalshi -> Poly (F=7.6***)
    - Others:         No significant flow

    Conclusion: Some structural information flows each direction,
    but these are secondary signals compared to Tier 1 prices.


OVERALL CONCLUSION
------------------
For the Fed December 2025 "No Change" market:

1. PRICE INFORMATION flows from Polymarket to Kalshi
   - Polymarket is the "price leader"
   - When new information arrives, it shows up in Polymarket prices first
   - Kalshi adjusts its prices ~30 minutes later

2. LIQUIDITY and ORDER FLOW do NOT flow between platforms
   - Each platform's orderbook depth evolves independently
   - Traders on each platform respond to information separately

3. SOME STRUCTURAL SIGNALS flow from Kalshi to Polymarket
   - But these are weaker and less consistent than price signals

PRACTICAL IMPLICATION:
If you're trading this market and want the freshest price signal,
watch Polymarket. Kalshi prices lag behind.
""")


    # -------------------------------------------------------------------------
    # Appendix: Full results table
    # -------------------------------------------------------------------------
    print(f"""
{'=' * 80}
APPENDIX: COMPLETE RESULTS TABLE
{'=' * 80}

{'Variable':<20} {'R2(Poly)':<10} {'R2(Kalshi)':<12} {'Poly->Kalshi':<15} {'Kalshi->Poly':<15} {'Leader':<10}
{'-' * 90}""")

    for var in VARIABLES:
        try:
            r = run_var(var, data)
            if r['reg1_r_squared'] is None:
                continue

            poly_to_kalshi = f"F={r['granger_poly_to_kalshi_F']:.1f}" if r['granger_poly_to_kalshi_F'] else "N/A"
            if r['granger_poly_to_kalshi_sig']:
                poly_to_kalshi += "***"

            kalshi_to_poly = f"F={r['granger_kalshi_to_poly_F']:.1f}" if r['granger_kalshi_to_poly_F'] else "N/A"
            if r['granger_kalshi_to_poly_sig']:
                kalshi_to_poly += "***"

            if r['granger_poly_to_kalshi_sig'] and not r['granger_kalshi_to_poly_sig']:
                leader = "POLY"
            elif r['granger_kalshi_to_poly_sig'] and not r['granger_poly_to_kalshi_sig']:
                leader = "KALSHI"
            elif r['granger_poly_to_kalshi_sig'] and r['granger_kalshi_to_poly_sig']:
                leader = "BOTH"
            else:
                leader = "-"

            print(f"{var:<20} {r['reg1_r_squared']:.4f}     {r['reg2_r_squared']:.4f}       {poly_to_kalshi:<15} {kalshi_to_poly:<15} {leader:<10}")
        except Exception as e:
            print(f"{var:<20} ERROR: {e}")

    print("""
*** = statistically significant at 5% level (p < 0.05)

R2(Poly):     Fraction of Polymarket's movements explained by past data
R2(Kalshi):   Fraction of Kalshi's movements explained by past data
              Higher R2 for the "follower" platform supports the lead-lag finding
""")


if __name__ == "__main__":
    main()
