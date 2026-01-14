#!/usr/bin/env python3
"""
Fed Decision December 2025 - No Change Market Orderbook Fetcher

This script fetches historical orderbook bid data for the "No Change" outcome
of the Fed December 2025 rate decision from both Polymarket and Kalshi.

MARKETS TRACKED:
- Polymarket: "No change in Fed interest rates after December 2025 meeting?"
  - YES token: Betting the Fed will NOT change rates
  - NO token: Betting the Fed WILL change rates

- Kalshi: KXFEDDECISION-25DEC-H0 (Hold 0 bps / No Change)
  - YES bids: Betting the Fed will NOT change rates
  - NO bids: Betting the Fed WILL change rates

APPROACH:
1. First fetch all Polymarket orderbook snapshots for the time range
   (Polymarket has ~20k snapshots for the 20-day period)
2. For each Polymarket timestamp, query Kalshi for the closest snapshot
   (Kalshi has millions of snapshots, so we align to Polymarket times)
3. Save aligned data with YES and NO bids for both platforms

OUTPUT FORMAT:
{
    "market_info": {...},
    "snapshots": [
        {
            "timestamp": 1763269232552,
            "polymarket": {
                "yes_bids": [...],
                "no_bids": [...]
            },
            "kalshi": {
                "yes_bids": [...],
                "yes_asks": [...],  # Derived from NO bids (price = 1 - no_bid_price)
                "no_bids": [...],
                "no_asks": [...]    # Derived from YES bids (price = 1 - yes_bid_price)
            }
        }
    ]
}

KALSHI ASK DERIVATION:
In Kalshi's binary market, YES + NO = $1.00. The orderbooks are mechanically linked:
- Selling YES at price P is equivalent to buying NO at price (1-P)
- Therefore: YES asks = NO bids with flipped prices
- And: NO asks = YES bids with flipped prices
This means we can derive the full orderbook from just the bid data.

RATE LIMITING:
- Uses 50 requests/second to stay under Dome API limits (500 per 10 sec)
- Parallel requests with batching for efficiency
"""

import requests
import json
import os
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# CONFIGURATION
# ==============================================================================

load_dotenv()
API_KEY = os.getenv("DOME_API_KEY")

# Time range: Nov 16 2025 12:00 AM ET to Dec 06 2025 11:59 PM ET
START_TIME_MS = 1763269200000
END_TIME_MS = 1765083599000

# Polymarket token IDs for "No Change" market
POLYMARKET_YES_TOKEN = "80746058984644290629624903019922696017323803605256698757445938534814122585786"
POLYMARKET_NO_TOKEN = "85970682092909806158465981896969493051954703765750954325716965011253694472992"

# Kalshi ticker for "No Change" (Hold 0 bps)
KALSHI_TICKER = "KXFEDDECISION-25DEC-H0"

# API endpoints
POLYMARKET_URL = "https://api.domeapi.io/v1/polymarket/orderbooks"
KALSHI_URL = "https://api.domeapi.io/v1/kalshi/orderbooks"

# Rate limiting: 50 requests per second (batch of 50, 1 second between batches)
BATCH_SIZE = 50

# Search window for Kalshi timestamp matching (±30 seconds)
KALSHI_SEARCH_WINDOW_MS = 30000


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def ms_to_et_string(ms):
    """Convert milliseconds timestamp to readable ET datetime string."""
    dt_utc = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    dt_et = dt_utc + timedelta(hours=-5)  # ET offset
    return dt_et.strftime("%b %d %Y %I:%M:%S %p")


def get_headers():
    """Return authorization headers for Dome API."""
    return {"Authorization": f"Bearer {API_KEY}"}


def extract_bids(snapshot, source="polymarket"):
    """
    Extract bids from a snapshot and sort by price (best bid first).

    For Polymarket: bids are in snapshot['bids'] as [{price, size}, ...]
    For Kalshi: bids are in snapshot['orderbook']['yes'] or ['no'] as [[price_cents, size], ...]
    """
    bids = []

    if source == "polymarket":
        raw_bids = snapshot.get("bids", [])
        for bid in raw_bids:
            bids.append({
                "price": bid["price"],
                "size": bid["size"]
            })
    elif source == "kalshi":
        # Kalshi returns [[price_cents, size], ...]
        raw_bids = snapshot
        for bid in raw_bids:
            price_cents = bid[0]
            size = bid[1]
            bids.append({
                "price": f"{price_cents / 100:.2f}",
                "size": str(size)
            })

    # Sort by price descending (best/highest bid first)
    bids.sort(key=lambda x: float(x["price"]), reverse=True)
    return bids


def derive_asks_from_bids(bids):
    """
    Derive asks from bids by flipping prices.

    In Kalshi's binary market, YES + NO = $1.00:
    - YES asks = NO bids with price flipped (ask_price = 1.00 - bid_price)
    - NO asks = YES bids with price flipped

    Asks are sorted ascending (best/lowest ask first).
    """
    asks = []
    for bid in bids:
        bid_price = float(bid["price"])
        ask_price = 1.00 - bid_price
        asks.append({
            "price": f"{ask_price:.2f}",
            "size": bid["size"]
        })
    # Sort by price ascending (best/lowest ask first)
    asks.sort(key=lambda x: float(x["price"]))
    return asks


# ==============================================================================
# POLYMARKET DATA FETCHING
# ==============================================================================

def fetch_polymarket_orderbooks(token_id, token_name):
    """
    Fetch all Polymarket orderbook snapshots for a token in the time range.
    Uses pagination to get all data.

    Returns list of snapshots with timestamps and bids.
    """
    print(f"\nFetching Polymarket {token_name} token orderbooks...", flush=True)

    headers = get_headers()
    all_snapshots = []
    pagination_key = None
    page = 1

    while True:
        # Build URL with pagination
        url = f"{POLYMARKET_URL}?limit=200&token_id={token_id}&start_time={START_TIME_MS}&end_time={END_TIME_MS}"
        if pagination_key:
            url += f"&pagination_key={pagination_key}"

        try:
            response = requests.get(url, headers=headers, timeout=30)
        except Exception as e:
            print(f"  Request error: {e}, retrying in 5s...", flush=True)
            time.sleep(5)
            continue

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 2)
            print(f"  Rate limited, waiting {retry_after}s...", flush=True)
            time.sleep(retry_after)
            continue

        # Handle 502/503/504 server errors with retry
        if response.status_code in [502, 503, 504]:
            print(f"  Server error {response.status_code}, retrying in 5s...", flush=True)
            time.sleep(5)
            continue

        if response.status_code != 200:
            print(f"  Error {response.status_code}: {response.text[:100]}", flush=True)
            break

        data = response.json()
        snapshots = data.get("snapshots", [])

        if not snapshots:
            break

        # Extract timestamp and bids from each snapshot
        for snap in snapshots:
            all_snapshots.append({
                "timestamp": snap.get("timestamp"),
                "bids": extract_bids(snap, source="polymarket")
            })

        # Progress update
        if page % 10 == 0:
            print(f"  Page {page}: {len(all_snapshots)} snapshots collected", flush=True)

        # Check for more pages
        pagination = data.get("pagination", {})
        if not pagination.get("has_more", False):
            break

        pagination_key = pagination.get("pagination_key")
        if not pagination_key:
            break

        page += 1

    print(f"  Completed: {len(all_snapshots)} {token_name} snapshots", flush=True)
    return all_snapshots


# ==============================================================================
# KALSHI DATA FETCHING
# ==============================================================================

def fetch_kalshi_at_timestamp(target_ts, headers, retries=3):
    """
    Fetch Kalshi orderbook snapshot closest to target timestamp.
    Returns YES bids and NO bids, or None if not found.

    Kalshi API returns:
    - orderbook.yes: [[price_cents, size], ...] - bids to buy YES
    - orderbook.no: [[price_cents, size], ...] - bids to buy NO
    """
    start_time = target_ts - KALSHI_SEARCH_WINDOW_MS
    end_time = target_ts + KALSHI_SEARCH_WINDOW_MS

    url = f"{KALSHI_URL}?limit=200&ticker={KALSHI_TICKER}&start_time={start_time}&end_time={end_time}"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 429:
                retry_after = response.json().get("retry_after", 2)
                time.sleep(retry_after)
                continue

            if response.status_code != 200:
                time.sleep(0.5)
                continue

            data = response.json()
            snapshots = data.get("snapshots", [])

            if not snapshots:
                time.sleep(0.5)
                continue

            # Find closest snapshot to target timestamp
            closest = min(snapshots, key=lambda s: abs(s.get("timestamp", 0) - target_ts))
            orderbook = closest.get("orderbook", {})

            return {
                "timestamp": closest.get("timestamp"),
                "yes_bids": extract_bids(orderbook.get("yes", []), source="kalshi"),
                "no_bids": extract_bids(orderbook.get("no", []), source="kalshi")
            }

        except Exception:
            time.sleep(0.5)
            continue

    return None


def fetch_kalshi_for_timestamps(timestamps):
    """
    Fetch Kalshi data for a list of Polymarket timestamps.
    Uses parallel requests with rate limiting (50 per second).

    Returns dict mapping timestamp -> kalshi data
    """
    print(f"\nFetching Kalshi orderbooks for {len(timestamps)} timestamps...", flush=True)
    print(f"  Rate: {BATCH_SIZE} requests/second", flush=True)

    headers = get_headers()
    results = {}
    completed = 0
    found = 0

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        # Process in batches of BATCH_SIZE, one batch per second
        for batch_start in range(0, len(timestamps), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(timestamps))
            batch = timestamps[batch_start:batch_end]
            batch_start_time = time.time()

            # Submit batch of requests
            futures = {
                executor.submit(fetch_kalshi_at_timestamp, ts, headers): ts
                for ts in batch
            }

            # Collect results
            for future in as_completed(futures):
                ts = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[ts] = result
                        found += 1
                except Exception:
                    pass
                completed += 1

            # Progress update every 500 requests
            if completed % 500 == 0:
                pct = (completed / len(timestamps)) * 100
                print(f"  Progress: {completed}/{len(timestamps)} ({pct:.1f}%) - found: {found}", flush=True)

            # Rate limit: ensure at least 1 second per batch
            elapsed = time.time() - batch_start_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

    print(f"  Completed: {found}/{len(timestamps)} Kalshi snapshots found", flush=True)
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 60)
    print("Fed December 2025 - No Change Market Orderbook Fetcher")
    print("=" * 60)
    print(f"\nTime range: {ms_to_et_string(START_TIME_MS)} to {ms_to_et_string(END_TIME_MS)} ET")
    print(f"\nMarkets:")
    print(f"  Polymarket YES token: {POLYMARKET_YES_TOKEN[:20]}...")
    print(f"  Polymarket NO token:  {POLYMARKET_NO_TOKEN[:20]}...")
    print(f"  Kalshi ticker:        {KALSHI_TICKER}")

    # -------------------------------------------------------------------------
    # Step 1: Fetch Polymarket YES orderbooks (to get timestamps)
    # -------------------------------------------------------------------------
    poly_yes = fetch_polymarket_orderbooks(POLYMARKET_YES_TOKEN, "YES")

    if not poly_yes:
        print("ERROR: No Polymarket YES data fetched")
        return

    # -------------------------------------------------------------------------
    # Step 2: Fetch Polymarket NO orderbooks
    # -------------------------------------------------------------------------
    poly_no = fetch_polymarket_orderbooks(POLYMARKET_NO_TOKEN, "NO")

    # Create lookup dict for NO bids by timestamp
    poly_no_by_ts = {snap["timestamp"]: snap["bids"] for snap in poly_no}

    # -------------------------------------------------------------------------
    # Step 3: Fetch Kalshi data at Polymarket timestamps
    # -------------------------------------------------------------------------
    timestamps = [snap["timestamp"] for snap in poly_yes]
    kalshi_data = fetch_kalshi_for_timestamps(timestamps)

    # -------------------------------------------------------------------------
    # Step 4: Combine all data into aligned snapshots
    # -------------------------------------------------------------------------
    print("\nCombining data into aligned snapshots...", flush=True)

    combined_snapshots = []

    for poly_snap in poly_yes:
        ts = poly_snap["timestamp"]

        snapshot = {
            "timestamp": ts,
            "timestamp_et": ms_to_et_string(ts),
            "polymarket": {
                "yes_bids": poly_snap["bids"],
                "no_bids": poly_no_by_ts.get(ts, [])
            },
            "kalshi": {
                "yes_bids": [],
                "yes_asks": [],  # Derived from NO bids
                "no_bids": [],
                "no_asks": []    # Derived from YES bids
            }
        }

        # Add Kalshi data if found
        if ts in kalshi_data:
            k = kalshi_data[ts]
            snapshot["kalshi"]["yes_bids"] = k["yes_bids"]
            snapshot["kalshi"]["no_bids"] = k["no_bids"]
            # Derive asks from opposite side's bids
            snapshot["kalshi"]["yes_asks"] = derive_asks_from_bids(k["no_bids"])
            snapshot["kalshi"]["no_asks"] = derive_asks_from_bids(k["yes_bids"])
            snapshot["kalshi_timestamp"] = k["timestamp"]
            snapshot["kalshi_time_diff_ms"] = k["timestamp"] - ts

        combined_snapshots.append(snapshot)

    # -------------------------------------------------------------------------
    # Step 5: Build output and save
    # -------------------------------------------------------------------------
    output = {
        "market_info": {
            "event": "Fed December 2025 Rate Decision",
            "outcome": "No Change (Hold 0 bps)",
            "description": "Orderbook bids for YES (no rate change) and NO (rate change) on both platforms",
            "time_range": {
                "start_ms": START_TIME_MS,
                "end_ms": END_TIME_MS,
                "start_et": ms_to_et_string(START_TIME_MS),
                "end_et": ms_to_et_string(END_TIME_MS)
            },
            "polymarket": {
                "yes_token_id": POLYMARKET_YES_TOKEN,
                "no_token_id": POLYMARKET_NO_TOKEN
            },
            "kalshi": {
                "ticker": KALSHI_TICKER
            },
            "data_note": "Polymarket: bids only. Kalshi: bids + derived asks (YES asks from NO bids, NO asks from YES bids). Timestamps aligned to Polymarket."
        },
        "total_snapshots": len(combined_snapshots),
        "kalshi_snapshots_found": len(kalshi_data),
        "kalshi_snapshots_missing": len(combined_snapshots) - len(kalshi_data),
        "snapshots": combined_snapshots
    }

    # Save to file
    output_file = "results/no_change/fed_nochange_orderbooks.json"
    os.makedirs("results/no_change", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total aligned snapshots: {len(combined_snapshots)}")
    print(f"Kalshi data found: {len(kalshi_data)}/{len(combined_snapshots)}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
