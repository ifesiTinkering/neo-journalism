#!/usr/bin/env python3
"""
Comprehensive Orderbook Preprocessing for Price Discovery Analysis

Extracts all relevant variables from raw orderbook data at 30-minute intervals:
- Tier 1: Mid price, Spread, Best bids
- Tier 2: Depth at best, Depth top 3, Total depth
- Tier 3: Order imbalance measures
- Tier 4: Number of levels, VWAP

Output: CSV ready for VAR analysis, Granger causality tests, etc.
"""

import json
import csv
from datetime import datetime, timezone, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "results/no_change/fed_nochange_orderbooks.json"
OUTPUT_FILE = "results/no_change/fed_nochange_30min_preprocessed.csv"
INTERVAL_MS = 30 * 60 * 1000  # 30 minutes


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ms_to_et_string(ms):
    """Convert milliseconds timestamp to ET datetime string."""
    dt_utc = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    dt_et = dt_utc + timedelta(hours=-5)
    return dt_et.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value, default=None):
    """Safely convert to float."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def get_best_bid(orders):
    """Get best (highest) bid price and size."""
    if orders and len(orders) > 0:
        return safe_float(orders[0]["price"]), safe_float(orders[0]["size"])
    return None, None


def get_top_n_depth(orders, n=3):
    """Get sum of sizes at top N price levels."""
    if not orders:
        return 0
    return sum(safe_float(o["size"], 0) for o in orders[:n])


def get_total_depth(orders):
    """Get total depth (sum of all sizes)."""
    if not orders:
        return 0
    return sum(safe_float(o["size"], 0) for o in orders)


def get_num_levels(orders):
    """Get number of price levels in orderbook."""
    return len(orders) if orders else 0


def get_vwap(orders):
    """Calculate volume-weighted average price."""
    if not orders:
        return None
    total_value = 0
    total_size = 0
    for o in orders:
        price = safe_float(o["price"], 0)
        size = safe_float(o["size"], 0)
        total_value += price * size
        total_size += size
    return total_value / total_size if total_size > 0 else None


def get_depth_within_range(orders, best_price, range_cents=0.05):
    """Get total depth within X cents of best price."""
    if not orders or best_price is None:
        return 0
    threshold = best_price - range_cents
    return sum(safe_float(o["size"], 0) for o in orders
               if safe_float(o["price"], 0) >= threshold)


def calculate_imbalance(yes_size, no_size):
    """
    Calculate order imbalance: (YES - NO) / (YES + NO)
    Returns value between -1 and +1
    Positive = more YES buying pressure (bullish on outcome)
    Negative = more NO buying pressure (bearish on outcome)
    """
    total = (yes_size or 0) + (no_size or 0)
    if total == 0:
        return None
    return ((yes_size or 0) - (no_size or 0)) / total


def calculate_mid_price(best_bid_yes, best_bid_no):
    """
    Calculate mid price using the formula:
    Mid = (Best_Bid_YES + Implied_Ask_YES) / 2
    Where Implied_Ask_YES = 1 - Best_Bid_NO
    """
    if best_bid_yes is None or best_bid_no is None:
        return None
    implied_ask_yes = 1 - best_bid_no
    return (best_bid_yes + implied_ask_yes) / 2


def calculate_spread(best_bid_yes, best_bid_no):
    """
    Calculate bid-ask spread:
    Spread = Implied_Ask_YES - Best_Bid_YES
           = (1 - Best_Bid_NO) - Best_Bid_YES
    """
    if best_bid_yes is None or best_bid_no is None:
        return None
    implied_ask_yes = 1 - best_bid_no
    return implied_ask_yes - best_bid_yes


def find_closest_snapshot(target_ts, snapshots):
    """Find the snapshot closest to target timestamp."""
    return min(snapshots, key=lambda s: abs(s["timestamp"] - target_ts))


# =============================================================================
# EXTRACT VARIABLES FROM ONE SNAPSHOT
# =============================================================================

def extract_variables(snap):
    """Extract all variables from a single snapshot."""

    # --- POLYMARKET ---
    poly_yes_bids = snap["polymarket"]["yes_bids"]
    poly_no_bids = snap["polymarket"]["no_bids"]

    # Best bids
    poly_best_bid_yes, poly_depth_best_yes = get_best_bid(poly_yes_bids)
    poly_best_bid_no, poly_depth_best_no = get_best_bid(poly_no_bids)

    # Mid price and spread
    poly_mid = calculate_mid_price(poly_best_bid_yes, poly_best_bid_no)
    poly_spread = calculate_spread(poly_best_bid_yes, poly_best_bid_no)

    # Depth measures
    poly_depth_top3_yes = get_top_n_depth(poly_yes_bids, 3)
    poly_depth_top3_no = get_top_n_depth(poly_no_bids, 3)
    poly_total_depth_yes = get_total_depth(poly_yes_bids)
    poly_total_depth_no = get_total_depth(poly_no_bids)

    # Depth within 5 cents
    poly_depth_5c_yes = get_depth_within_range(poly_yes_bids, poly_best_bid_yes, 0.05)
    poly_depth_5c_no = get_depth_within_range(poly_no_bids, poly_best_bid_no, 0.05)

    # Number of levels
    poly_num_levels_yes = get_num_levels(poly_yes_bids)
    poly_num_levels_no = get_num_levels(poly_no_bids)

    # VWAP
    poly_vwap_yes = get_vwap(poly_yes_bids)
    poly_vwap_no = get_vwap(poly_no_bids)

    # Imbalances
    poly_imbalance_best = calculate_imbalance(poly_depth_best_yes, poly_depth_best_no)
    poly_imbalance_top3 = calculate_imbalance(poly_depth_top3_yes, poly_depth_top3_no)
    poly_imbalance_total = calculate_imbalance(poly_total_depth_yes, poly_total_depth_no)


    # --- KALSHI ---
    kalshi_yes_bids = snap["kalshi"]["yes_bids"]
    kalshi_no_bids = snap["kalshi"]["no_bids"]

    # Best bids
    kalshi_best_bid_yes, kalshi_depth_best_yes = get_best_bid(kalshi_yes_bids)
    kalshi_best_bid_no, kalshi_depth_best_no = get_best_bid(kalshi_no_bids)

    # Mid price and spread
    kalshi_mid = calculate_mid_price(kalshi_best_bid_yes, kalshi_best_bid_no)
    kalshi_spread = calculate_spread(kalshi_best_bid_yes, kalshi_best_bid_no)

    # Depth measures
    kalshi_depth_top3_yes = get_top_n_depth(kalshi_yes_bids, 3)
    kalshi_depth_top3_no = get_top_n_depth(kalshi_no_bids, 3)
    kalshi_total_depth_yes = get_total_depth(kalshi_yes_bids)
    kalshi_total_depth_no = get_total_depth(kalshi_no_bids)

    # Depth within 5 cents
    kalshi_depth_5c_yes = get_depth_within_range(kalshi_yes_bids, kalshi_best_bid_yes, 0.05)
    kalshi_depth_5c_no = get_depth_within_range(kalshi_no_bids, kalshi_best_bid_no, 0.05)

    # Number of levels
    kalshi_num_levels_yes = get_num_levels(kalshi_yes_bids)
    kalshi_num_levels_no = get_num_levels(kalshi_no_bids)

    # VWAP
    kalshi_vwap_yes = get_vwap(kalshi_yes_bids)
    kalshi_vwap_no = get_vwap(kalshi_no_bids)

    # Imbalances
    kalshi_imbalance_best = calculate_imbalance(kalshi_depth_best_yes, kalshi_depth_best_no)
    kalshi_imbalance_top3 = calculate_imbalance(kalshi_depth_top3_yes, kalshi_depth_top3_no)
    kalshi_imbalance_total = calculate_imbalance(kalshi_total_depth_yes, kalshi_total_depth_no)


    return {
        # Timestamp
        "timestamp_ms": snap["timestamp"],
        "timestamp_et": ms_to_et_string(snap["timestamp"]),

        # ===================
        # POLYMARKET
        # ===================

        # Tier 1: Price measures
        "poly_mid": poly_mid,
        "poly_spread": poly_spread,
        "poly_best_bid_yes": poly_best_bid_yes,
        "poly_best_bid_no": poly_best_bid_no,

        # Tier 2: Depth measures
        "poly_depth_best_yes": poly_depth_best_yes,
        "poly_depth_best_no": poly_depth_best_no,
        "poly_depth_top3_yes": poly_depth_top3_yes,
        "poly_depth_top3_no": poly_depth_top3_no,
        "poly_total_depth_yes": poly_total_depth_yes,
        "poly_total_depth_no": poly_total_depth_no,
        "poly_depth_5c_yes": poly_depth_5c_yes,
        "poly_depth_5c_no": poly_depth_5c_no,

        # Tier 3: Imbalance measures
        "poly_imbalance_best": poly_imbalance_best,
        "poly_imbalance_top3": poly_imbalance_top3,
        "poly_imbalance_total": poly_imbalance_total,

        # Tier 4: Book shape
        "poly_num_levels_yes": poly_num_levels_yes,
        "poly_num_levels_no": poly_num_levels_no,
        "poly_vwap_yes": poly_vwap_yes,
        "poly_vwap_no": poly_vwap_no,

        # ===================
        # KALSHI
        # ===================

        # Tier 1: Price measures
        "kalshi_mid": kalshi_mid,
        "kalshi_spread": kalshi_spread,
        "kalshi_best_bid_yes": kalshi_best_bid_yes,
        "kalshi_best_bid_no": kalshi_best_bid_no,

        # Tier 2: Depth measures
        "kalshi_depth_best_yes": kalshi_depth_best_yes,
        "kalshi_depth_best_no": kalshi_depth_best_no,
        "kalshi_depth_top3_yes": kalshi_depth_top3_yes,
        "kalshi_depth_top3_no": kalshi_depth_top3_no,
        "kalshi_total_depth_yes": kalshi_total_depth_yes,
        "kalshi_total_depth_no": kalshi_total_depth_no,
        "kalshi_depth_5c_yes": kalshi_depth_5c_yes,
        "kalshi_depth_5c_no": kalshi_depth_5c_no,

        # Tier 3: Imbalance measures
        "kalshi_imbalance_best": kalshi_imbalance_best,
        "kalshi_imbalance_top3": kalshi_imbalance_top3,
        "kalshi_imbalance_total": kalshi_imbalance_total,

        # Tier 4: Book shape
        "kalshi_num_levels_yes": kalshi_num_levels_yes,
        "kalshi_num_levels_no": kalshi_num_levels_no,
        "kalshi_vwap_yes": kalshi_vwap_yes,
        "kalshi_vwap_no": kalshi_vwap_no,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Orderbook Preprocessing for Price Discovery Analysis")
    print("=" * 60)

    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    snapshots = data["snapshots"]
    print(f"Loaded {len(snapshots)} snapshots")

    # Generate 30-minute interval timestamps
    start_ts = snapshots[0]["timestamp"]
    end_ts = snapshots[-1]["timestamp"]

    interval_timestamps = []
    current_ts = start_ts
    while current_ts <= end_ts:
        interval_timestamps.append(current_ts)
        current_ts += INTERVAL_MS

    print(f"\nExtracting {len(interval_timestamps)} samples at 30-minute intervals...")

    # Extract variables for each interval
    rows = []
    for i, target_ts in enumerate(interval_timestamps):
        snap = find_closest_snapshot(target_ts, snapshots)
        row = extract_variables(snap)
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(interval_timestamps)}")

    # =========================================================================
    # STEP 1: Compute Delta (Δ) columns - change from previous period
    # Δvar_t = var_t - var_{t-1}
    # =========================================================================
    print(f"\nComputing delta (Δ) columns...")

    # List of numeric columns to compute deltas for (exclude timestamps)
    numeric_cols = [col for col in rows[0].keys()
                    if col not in ["timestamp_ms", "timestamp_et"]]

    for i in range(len(rows)):
        for col in numeric_cols:
            delta_col = f"d_{col}"  # d_ prefix for delta
            if i == 0:
                # First row has no previous period
                rows[i][delta_col] = None
            else:
                curr_val = rows[i][col]
                prev_val = rows[i-1][col]
                if curr_val is not None and prev_val is not None:
                    rows[i][delta_col] = curr_val - prev_val
                else:
                    rows[i][delta_col] = None

    # =========================================================================
    # STEP 2: Compute Lagged Delta columns - previous period's delta
    # Δvar_lag1 = Δvar_{t-1}
    # =========================================================================
    print(f"Computing lagged delta (Δ_lag1) columns...")

    delta_cols = [f"d_{col}" for col in numeric_cols]

    for i in range(len(rows)):
        for delta_col in delta_cols:
            lag_col = f"{delta_col}_lag1"  # _lag1 suffix for lag
            if i <= 1:
                # First two rows don't have lagged deltas
                rows[i][lag_col] = None
            else:
                rows[i][lag_col] = rows[i-1][delta_col]

    # =========================================================================
    # Remove first 2 rows (no complete lag data)
    # =========================================================================
    print(f"Removing first 2 rows (incomplete lag data)...")
    rows_complete = rows[2:]
    print(f"  Rows before: {len(rows)}, after: {len(rows_complete)}")

    # Write to CSV
    print(f"\nWriting to {OUTPUT_FILE}...")
    fieldnames = list(rows_complete[0].keys())

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_complete)

    # Summary
    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(rows_complete)}")
    print(f"Columns: {len(fieldnames)}")

    # Count column types
    base_cols = [c for c in fieldnames if not c.startswith("d_")]
    delta_cols = [c for c in fieldnames if c.startswith("d_") and not c.endswith("_lag1")]
    lag_cols = [c for c in fieldnames if c.endswith("_lag1")]

    print(f"\nColumn breakdown:")
    print(f"  Base variables: {len(base_cols)} (original values)")
    print(f"  Delta (Δ) columns: {len(delta_cols)} (change from t-1)")
    print(f"  Lagged delta columns: {len(lag_cols)} (Δ from t-1)")

    # Show column categories
    print(f"\nVariables extracted:")
    print(f"  Tier 1 (Price):     mid, spread, best_bid_yes, best_bid_no")
    print(f"  Tier 2 (Depth):     depth_best, depth_top3, total_depth, depth_5c")
    print(f"  Tier 3 (Imbalance): imbalance_best, imbalance_top3, imbalance_total")
    print(f"  Tier 4 (Shape):     num_levels, vwap")

    # Show sample data
    print(f"\n{'=' * 60}")
    print("SAMPLE DATA (First 3 rows)")
    print(f"{'=' * 60}")

    for i, row in enumerate(rows[:3]):
        print(f"\n--- Row {i + 1}: {row['timestamp_et']} ---")
        print(f"  Poly:  mid={row['poly_mid']:.4f}, spread={row['poly_spread']:.4f}, "
              f"imbalance={row['poly_imbalance_best']:.3f}" if row['poly_imbalance_best'] else "")
        print(f"  Kalshi: mid={row['kalshi_mid']:.4f}, spread={row['kalshi_spread']:.4f}, "
              f"imbalance={row['kalshi_imbalance_best']:.3f}" if row['kalshi_imbalance_best'] else "")
        print(f"  Poly depth:  best_yes={row['poly_depth_best_yes']:.0f}, "
              f"best_no={row['poly_depth_best_no']:.0f}, "
              f"total_yes={row['poly_total_depth_yes']:.0f}")
        print(f"  Kalshi depth: best_yes={row['kalshi_depth_best_yes']:.0f}, "
              f"best_no={row['kalshi_depth_best_no']:.0f}, "
              f"total_yes={row['kalshi_total_depth_yes']:.0f}")


if __name__ == "__main__":
    main()
