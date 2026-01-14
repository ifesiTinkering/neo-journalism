# Fed December 2025 Rate Decision: Cross-Platform Information Flow Analysis

## Overview

This repository analyzes **how information flows between Kalshi and Polymarket** for the Federal Reserve December 2025 interest rate decision market. Rather than just looking at price discovery, we examine multiple dimensions of the orderbook to understand which platform reacts first to new information across:

- **Price signals** - Do price changes appear on one platform first?
- **Liquidity depth** - Do liquidity shifts propagate across platforms?
- **Order imbalances** - Does buying/selling pressure transfer between markets?
- **Book structure** - Do structural changes (levels, concentration) lead or lag?

**Market Question:** Will the Fed cut rates at the December 9-10, 2025 FOMC meeting?

**Outcomes Traded:**
- 25 bps decrease (cut)
- No change (hold)
- 50+ bps decrease (aggressive cut)
- 25+ bps increase (hike)

**Key Findings:**
1. **Price information** flows from Polymarket to Kalshi (Poly leads by ~30 min)
2. **Liquidity and order flow** do NOT flow between platforms (independent dynamics)
3. **Some structural signals** flow from Kalshi to Polymarket (weaker, less consistent)

---

## Quick Start

### Prerequisites

```bash
pip install requests python-dotenv scipy
```

You need a [Dome API](https://domeapi.io) key. Create a `.env` file:

```
DOME_API_KEY=your_api_key_here
```

### Running the Analysis

```bash
# Step 1: Fetch orderbook data from Polymarket and Kalshi
python fetch_fed_nochange_orderbooks.py

# Step 2: Preprocess into time series with all variables
python preprocess_orderbooks.py

# Step 3: Run VAR analysis for price discovery
python run_var.py
```

### Output Files

```
results/no_change/
├── fed_nochange_orderbooks.json        # Raw orderbook snapshots (~275MB)
├── fed_nochange_30min_preprocessed.csv # Preprocessed time series (1006 rows, 116 columns)
└── var_results.md                      # Complete analysis with explanations
```

---

## Methodology

### Data Collection

The analysis uses orderbook snapshots from both platforms over a 20-day window (Nov 16 - Dec 6, 2025). Polymarket snapshots are fetched first (~20,000 snapshots), then Kalshi orderbooks are queried at matching timestamps.

### Preprocessed Variables (4 Tiers)

For each platform (Polymarket and Kalshi), we extract 19 orderbook variables organized into 4 tiers:

#### Tier 1: Price Signals
| Variable | Formula | Description |
|----------|---------|-------------|
| `mid` | (Best_Bid_YES + (1 - Best_Bid_NO)) / 2 | Mid-market price |
| `spread` | (1 - Best_Bid_NO) - Best_Bid_YES | Bid-ask spread |
| `best_bid_yes` | Top bid price for YES | Best YES bid |
| `best_bid_no` | Top bid price for NO | Best NO bid |

#### Tier 2: Depth Measures
| Variable | Formula | Description |
|----------|---------|-------------|
| `depth_best_yes` | Size at best YES bid | Liquidity at top of book (YES) |
| `depth_best_no` | Size at best NO bid | Liquidity at top of book (NO) |
| `depth_top3_yes` | Sum of top 3 YES bid sizes | Near-market liquidity (YES) |
| `depth_top3_no` | Sum of top 3 NO bid sizes | Near-market liquidity (NO) |
| `total_depth_yes` | Sum of all YES bid sizes | Total book depth (YES) |
| `total_depth_no` | Sum of all NO bid sizes | Total book depth (NO) |

#### Tier 3: Imbalance Measures
| Variable | Formula | Description |
|----------|---------|-------------|
| `imbalance_best` | (depth_best_yes - depth_best_no) / (depth_best_yes + depth_best_no) | Order imbalance at best price |
| `imbalance_top3` | (depth_top3_yes - depth_top3_no) / (depth_top3_yes + depth_top3_no) | Order imbalance in top 3 levels |
| `imbalance_total` | (total_depth_yes - total_depth_no) / (total_depth_yes + total_depth_no) | Total book imbalance |

#### Tier 4: Book Shape
| Variable | Formula | Description |
|----------|---------|-------------|
| `depth_5c_yes` | Sum of YES bids within 5 cents of best | Concentrated liquidity (YES) |
| `depth_5c_no` | Sum of NO bids within 5 cents of best | Concentrated liquidity (NO) |
| `num_levels_yes` | Count of YES price levels | Book depth structure (YES) |
| `num_levels_no` | Count of NO price levels | Book depth structure (NO) |
| `vwap_yes` | Volume-weighted average price (YES) | Average execution price (YES) |
| `vwap_no` | Volume-weighted average price (NO) | Average execution price (NO) |

### VAR Analysis

Vector Autoregression (VAR) runs two regressions to test lead-lag relationships:

```
Regression 1: Delta_Poly_t   = a1 + b11*Delta_Poly_{t-1}   + b12*Delta_Kalshi_{t-1}
Regression 2: Delta_Kalshi_t = a2 + b21*Delta_Poly_{t-1}   + b22*Delta_Kalshi_{t-1}
```

Granger causality tests determine significance:
- If b21 is significant: Polymarket leads (Poly changes predict Kalshi changes)
- If b12 is significant: Kalshi leads (Kalshi changes predict Poly changes)

---

## Results

### Summary by Tier

| Tier | Question | Finding |
|------|----------|---------|
| **Tier 1: Price Signals** | When prices move, which platform moves first? | **POLYMARKET LEADS** |
| **Tier 2: Liquidity Depth** | When liquidity shifts, which platform shows it first? | No cross-platform flow |
| **Tier 3: Order Imbalances** | When buying/selling pressure builds, where first? | No cross-platform flow |
| **Tier 4: Book Structure** | When orderbook structure changes, which leads? | Mixed (weak signals both ways) |

### Tier 1: Price Signals (Primary Finding)

```
Mid Price:
    Poly -> Kalshi:  F = 112.2, p = 0.0000 (SIGNIFICANT)
    Kalshi -> Poly:  F = 0.4,   p = 0.5490 (not significant)

Best Bid YES:
    Poly -> Kalshi:  F = 104.9, p = 0.0000 (SIGNIFICANT)

Best Bid NO:
    Poly -> Kalshi:  F = 133.6, p = 0.0000 (SIGNIFICANT)
```

A 1% move in Polymarket mid price predicts a 0.51% move in Kalshi mid price in the next 30-minute period. Price information consistently flows from Polymarket to Kalshi.

### All Variables Summary

| Variable | Leader | Interpretation |
|----------|--------|----------------|
| mid | POLY | Poly moves -> Kalshi follows (0.51x) |
| best_bid_yes | POLY | Poly moves -> Kalshi follows (0.45x) |
| best_bid_no | POLY | Poly moves -> Kalshi follows (0.55x) |
| num_levels_yes | POLY | Poly moves -> Kalshi follows (0.06x) |
| depth_5c_no | KALSHI | Kalshi moves -> Poly follows (-0.20x) |
| vwap_no | KALSHI | Kalshi moves -> Poly follows (-0.05x) |
| All others | - | No significant cross-platform flow |

See `results/no_change/var_results.md` for the complete analysis with detailed explanations.

---

## Adapting for Other Markets

### Changing the Market

To analyze a different market, modify the configuration in `fetch_fed_nochange_orderbooks.py`:

```python
# Polymarket token IDs (find these in the Polymarket API or URL)
POLYMARKET_YES_TOKEN = "your_yes_token_id"
POLYMARKET_NO_TOKEN = "your_no_token_id"

# Kalshi ticker (find this on kalshi.com market URL)
KALSHI_TICKER = "KXYOURMARKET-TICKER"

# Time range (milliseconds since epoch)
START_TIME_MS = 1763269200000  # Start timestamp
END_TIME_MS = 1765083599000    # End timestamp
```

### Finding Token IDs

**Polymarket:**
- Go to the market page on polymarket.com
- Open browser developer tools (F12)
- Look at network requests to find `token_id` values

**Kalshi:**
- The ticker is in the market URL: `kalshi.com/markets/TICKER/...`

### Example: Analyzing a Different Fed Decision

```python
# For "25 bps cut" outcome instead of "No change":
POLYMARKET_YES_TOKEN = "12345..."  # YES = Fed cuts 25 bps
POLYMARKET_NO_TOKEN = "67890..."   # NO = Fed does NOT cut 25 bps
KALSHI_TICKER = "KXFEDDECISION-25DEC-D25"  # Down 25 bps ticker
```

---

## File Structure

```
FED_DECISION/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .env                               # API key (not committed)
├── .gitignore                         # Excludes .env
│
├── fetch_fed_nochange_orderbooks.py   # Step 1: Fetch raw orderbook data
├── preprocess_orderbooks.py           # Step 2: Extract 19 variables + create lags
├── run_var.py                         # Step 3: VAR analysis (extensively documented)
│
└── results/no_change/
    ├── fed_nochange_orderbooks.json   # Raw orderbook snapshots
    ├── fed_nochange_30min_preprocessed.csv  # Processed time series (116 columns)
    └── var_results.md                 # Complete analysis with tier-by-tier results
```

---

## Market Context

### The November 2025 Volatility Window

Rate cut odds for December 2025 swung dramatically during November 12-21:

```
100% |
 95% | <- Mid-October baseline (~95-98%)
 90% | ------------
 85% |            \
 80% |             \
 75% |              \
 70% |               \  <- Nov 12-13: Bostic + Collins speeches
 65% |                \
 60% |                 \
 55% |                  \  <- Nov 14: Schmid Denver speech
 50% |                   ----  <- Nov 17: "No change" briefly favored
 45% |                       \
 40% |                        \  <- Nov 17 PM: Waller London speech
 35% |                    ------- <- Nov 19: FOMC minutes (hawkish)
 30% |                          ---- <- Nov 20: Jobs report
 25% |                              \
 22% |                               <- Nov 20: BOTTOM (22-30%)
     |                                 |
 35% |                              ---/
 50% |                            /
 65% |                          /
 73% |                        --- <- Nov 21 AM: Williams Chile speech
 79% |                      / <- Nov 21 EOD
 85% |-------------------------- <- Late Nov: stabilizes ~80-85%
 94% |------------------------------ <- Current (Dec 5)
     +----------------------------------------------
      Nov 12  13  14  15  16  17  18  19  20  21  22
```

### Key Catalysts

| Date | Event | Movement |
|------|-------|----------|
| Nov 17, 3:35 PM | Waller London speech | +5-10% (first dovish pushback) |
| Nov 19, 2:00 PM | FOMC Minutes release | -20% (biggest drop) |
| Nov 21, 8:22 AM | Williams Chile speech | +38% (biggest reversal) |

### Market Structure

**Polymarket**
- URL: https://polymarket.com/event/fed-decision-in-december
- Total Volume: ~$260 million

**Kalshi**
- URL: https://kalshi.com/markets/kxfeddecision/fed-meeting
- Volume: ~$15.8 million

---

## Data Sources

- [Dome API](https://domeapi.io) - Historical orderbook data
- [CME FedWatch Tool](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html) - Fed funds futures
- [Federal Reserve Speeches](https://www.federalreserve.gov/newsevents/speeches.htm) - Official statements

---

## License

MIT License - see LICENSE file
