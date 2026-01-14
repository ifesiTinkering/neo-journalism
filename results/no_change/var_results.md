================================================================================
CROSS-PLATFORM INFORMATION FLOW ANALYSIS
Event: Fed Decision in December 2025 | Market: No Change
================================================================================

Loading results/no_change/fed_nochange_30min_preprocessed.csv...
Loaded 1006 observations (30-minute intervals)


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

================================================================================
DETAILED ANALYSIS: MID PRICE (Primary Price Signal)
================================================================================

======================================================================
VAR RESULTS: MID
======================================================================
Observations: 1003

--- Regression 1: Predicting ΔPoly ---
ΔPoly_t = -0.000347 + 0.1594·ΔPoly_{t-1} + 0.0306·ΔKalshi_{t-1}
R² = 0.0339 (3.4% of variance explained)

--- Regression 2: Predicting ΔKalshi ---
ΔKalshi_t = -0.000349 + 0.5062·ΔPoly_{t-1} + -0.2990·ΔKalshi_{t-1}
R² = 0.1106 (11.1% of variance explained)

--- Granger Causality Tests ---
Kalshi → Poly:  F = 0.359, p = 0.5490 (not significant)
Poly → Kalshi:  F = 112.155, p = 0.0000 ***

--- Interpretation ---
→ POLYMARKET LEADS: Poly predicts Kalshi (p=0.0000)
  A 1% move in Poly predicts a 0.51% move in Kalshi next period

================================================================================
DETAILED EXPLANATION OF RESULTS: MID
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

    DeltaPoly_t = -0.000347 + 0.1594*DeltaPoly_{t-1} + 0.0306*DeltaKalshi_{t-1}
                  \_________/   \______________/                  \________________/
                   intercept    own lag effect                    CROSS-MARKET EFFECT
                   (alpha)      (beta_11)                         (beta_12)
                                                                  |
                                                                  v
                                            This is what we TEST: Does Kalshi predict Poly?
                                            beta_12 = 0.0306

Regression 2 (Predicting Kalshi):

    DeltaKalshi_t = -0.000349 + 0.5062*DeltaPoly_{t-1} + -0.2990*DeltaKalshi_{t-1}
                    \_________/   \______________/                  \________________/
                     intercept    CROSS-MARKET EFFECT               own lag effect
                     (alpha)      (beta_21)                         (beta_22)
                                  |
                                  v
                  This is what we TEST: Does Poly predict Kalshi?
                  beta_21 = 0.5062


WHAT THE BETA COEFFICIENTS MEAN
-------------------------------
beta_21 = 0.5062 (Poly lag in Kalshi regression)

    Interpretation: When Polymarket moves by 1 unit, Kalshi moves by 0.5062 units
                    in the NEXT period, on average.

    Example: If Poly's mid price increased by +2% last period,
             we'd expect Kalshi to increase by 0.5062 * 2% = 1.0124%
             this period (all else equal).

beta_12 = 0.0306 (Kalshi lag in Poly regression)

    Interpretation: When Kalshi moves by 1 unit, Poly moves by 0.0306 units
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

    F = 112.155
    p = 0.000000

    What F means: Removing Poly's lag made the Kalshi prediction 112.2x worse
                  relative to the baseline noise level.

    What p means: There's a 0.0000% chance we'd see an F this large
                  if Poly had NO real predictive power (pure chance).

    Decision: p < 0.05, so REJECT null hypothesis. Poly DOES predict Kalshi.

Testing "Does Kalshi predict Poly?" (Kalshi -> Poly):

    F = 0.359
    p = 0.549035

    What F means: Removing Kalshi's lag made the Poly prediction 0.4x worse
                  relative to the baseline noise level.

    What p means: There's a 54.90% chance we'd see an F this large
                  if Kalshi had NO real predictive power (pure chance).

    Decision: p >= 0.05, so CANNOT reject null. No evidence Kalshi predicts Poly.


R-SQUARED: HOW WELL DO THE MODELS FIT?
--------------------------------------
R-squared tells us what fraction of price movements we can explain.

    R²(Poly regression)   = 0.0339 = 3.4%
    R²(Kalshi regression) = 0.1106 = 11.1%

    Interpretation:
    - 3.4% of Poly's price movements can be explained by past data
    - 11.1% of Kalshi's price movements can be explained by past data

    The fact that R²(Kalshi) > R²(Poly) supports the finding that Kalshi is
    the "follower" - its movements are more predictable because it's reacting
    to Polymarket.


FINAL INTERPRETATION
--------------------

    POLYMARKET LEADS PRICE DISCOVERY

    Evidence:
    1. Poly -> Kalshi: F=112.2, p=0.0000 (SIGNIFICANT)
       When Poly moves, Kalshi follows next period.

    2. Kalshi -> Poly: F=0.4, p=0.5490 (not significant)
       When Kalshi moves, Poly does NOT follow.

    Magnitude: A 1% move in Poly predicts a 0.51% move in Kalshi.

    What this means in practice:
    - New information appears on Polymarket FIRST
    - Kalshi prices adjust ~30 minutes LATER
    - Polymarket has more informed/faster traders for this market


================================================================================
TIER 1: PRICE SIGNALS
Question: When prices move, which platform moves first?
================================================================================

Variable             Poly->Kalshi       Kalshi->Poly       Leader       Interpretation
----------------------------------------------------------------------------------------------------
mid                  F=112.2***         F=0.4              POLY         Poly moves -> Kalshi follows (0.51x)
spread               F=0.8              F=0.2              -            No significant cross-platform flow
best_bid_yes         F=104.9***         F=1.5              POLY         Poly moves -> Kalshi follows (0.45x)
best_bid_no          F=133.6***         F=0.6              POLY         Poly moves -> Kalshi follows (0.55x)

================================================================================
TIER 2: LIQUIDITY DEPTH
Question: When liquidity shifts, which platform shows it first?
================================================================================

Variable             Poly->Kalshi       Kalshi->Poly       Leader       Interpretation
----------------------------------------------------------------------------------------------------
depth_best_yes       F=0.0              F=0.0              -            No significant cross-platform flow
depth_best_no        F=2.4              F=0.6              -            No significant cross-platform flow
depth_top3_yes       F=1.1              F=0.7              -            No significant cross-platform flow
depth_top3_no        F=1.6              F=2.1              -            No significant cross-platform flow
total_depth_yes      F=0.0              F=2.3              -            No significant cross-platform flow
total_depth_no       F=0.1              F=2.1              -            No significant cross-platform flow

================================================================================
TIER 3: ORDER IMBALANCES
Question: When buying/selling pressure builds, where does it appear first?
================================================================================

Variable             Poly->Kalshi       Kalshi->Poly       Leader       Interpretation
----------------------------------------------------------------------------------------------------
imbalance_best       F=0.0              F=1.3              -            No significant cross-platform flow
imbalance_top3       F=0.5              F=0.2              -            No significant cross-platform flow
imbalance_total      F=1.1              F=1.4              -            No significant cross-platform flow

================================================================================
TIER 4: BOOK STRUCTURE
Question: When the orderbook structure changes, which platform leads?
================================================================================

Variable             Poly->Kalshi       Kalshi->Poly       Leader       Interpretation
----------------------------------------------------------------------------------------------------
depth_5c_yes         F=2.6              F=0.6              -            No significant cross-platform flow
depth_5c_no          F=0.3              F=4.5***           KALSHI       Kalshi moves -> Poly follows (-0.20x)
num_levels_yes       F=4.7***           F=2.0              POLY         Poly moves -> Kalshi follows (0.06x)
num_levels_no        F=0.8              F=0.1              -            No significant cross-platform flow
vwap_yes             F=0.1              F=0.3              -            No significant cross-platform flow
vwap_no              F=1.7              F=7.6***           KALSHI       Kalshi moves -> Poly follows (-0.05x)


================================================================================
SUMMARY OF FINDINGS
================================================================================

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


================================================================================
APPENDIX: COMPLETE RESULTS TABLE
================================================================================

Variable             R2(Poly)   R2(Kalshi)   Poly->Kalshi    Kalshi->Poly    Leader    
------------------------------------------------------------------------------------------
mid                  0.0339     0.1106       F=112.2***      F=0.4           POLY      
spread               0.1760     0.1837       F=0.8           F=0.2           -         
best_bid_yes         0.0231     0.1013       F=104.9***      F=1.5           POLY      
best_bid_no          0.0377     0.1197       F=133.6***      F=0.6           POLY      
depth_best_yes       0.2156     0.1295       F=0.0           F=0.0           -         
depth_best_no        0.0678     0.1711       F=2.4           F=0.6           -         
depth_top3_yes       0.0161     0.1949       F=1.1           F=0.7           -         
depth_top3_no        0.0779     0.0165       F=1.6           F=2.1           -         
total_depth_yes      0.0037     0.0004       F=0.0           F=2.3           -         
total_depth_no       0.1659     0.0020       F=0.1           F=2.1           -         
depth_5c_yes         0.0022     0.1161       F=2.6           F=0.6           -         
depth_5c_no          0.0798     0.0021       F=0.3           F=4.5***        KALSHI    
imbalance_best       0.1414     0.1435       F=0.0           F=1.3           -         
imbalance_top3       0.0991     0.1053       F=0.5           F=0.2           -         
imbalance_total      0.0349     0.0348       F=1.1           F=1.4           -         
num_levels_yes       0.0079     0.0379       F=4.7***        F=2.0           POLY      
num_levels_no        0.1856     0.1012       F=0.8           F=0.1           -         
vwap_yes             0.0268     0.0082       F=0.1           F=0.3           -         
vwap_no              0.0392     0.0083       F=1.7           F=7.6***        KALSHI    

*** = statistically significant at 5% level (p < 0.05)

R2(Poly):     Fraction of Polymarket's movements explained by past data
R2(Kalshi):   Fraction of Kalshi's movements explained by past data
              Higher R2 for the "follower" platform supports the lead-lag finding

