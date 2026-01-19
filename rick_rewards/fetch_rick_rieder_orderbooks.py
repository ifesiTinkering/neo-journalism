#!/usr/bin/env python3
"""
Rick Rieder Fed Chair Nomination Market - Orderbook Fetcher

Fetches raw orderbook data from Polymarket via Dome API for the market:
"Will Trump nominate Rick Rieder as the next Fed chair?"

This script collects raw orderbook snapshots without preprocessing.
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

# Polymarket token IDs for Rick Rieder market
POLYMARKET_YES_TOKEN = "16206267440377108972343351482425564384973031696941663558076310969498822538172"
POLYMARKET_NO_TOKEN = "97278843093518170983116788293559302501059038797143772070255096806379425818363"

# API endpoint
POLYMARKET_URL = "https://api.domeapi.io/v1/polymarket/orderbooks"

# Retry configuration
MAX_RETRIES = 4

# Rate limiting: 50 requests/second = 0.02s between requests
# Using 0.025s to be slightly conservative
REQUEST_DELAY = 0.025


# ==============================================================================
# TIMESTAMP UTILITIES
# ==============================================================================

def et_to_unix_ms(year, month, day, hour=0, minute=0, second=0):
    """
    Convert Eastern Time date/time to Unix timestamp in milliseconds.

    Example:
        et_to_unix_ms(2026, 1, 11, 0, 0, 0)  -> Start of Jan 11, 2026 ET in ms
    """
    # Create datetime in UTC, then adjust for ET offset (-5 hours)
    # ET midnight = UTC 5:00 AM
    dt_et = datetime(year, month, day, hour, minute, second)
    # ET is UTC-5 (ignoring DST for simplicity - January is standard time)
    dt_utc = dt_et + timedelta(hours=5)
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return int(dt_utc.timestamp() * 1000)


def unix_ms_to_et_string(ms):
    """Convert Unix milliseconds to readable ET datetime string."""
    dt_utc = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    dt_et = dt_utc - timedelta(hours=5)  # UTC to ET
    return dt_et.strftime("%b %d %Y %I:%M:%S %p ET")


# Time range: Jan 11, 2026 start of day to Jan 15, 2026 midnight ET
START_TIME_MS = et_to_unix_ms(2026, 1, 11, 0, 0, 0)   # Jan 11, 2026 00:00:00 ET
END_TIME_MS = et_to_unix_ms(2026, 1, 16, 0, 0, 0)     # Jan 16, 2026 00:00:00 ET (= end of Jan 15)


# ==============================================================================
# API FUNCTIONS
# ==============================================================================

def get_headers():
    """Return authorization headers for Dome API."""
    return {"Authorization": f"Bearer {API_KEY}"}


def fetch_with_retries(url, headers, max_retries=MAX_RETRIES):
    """
    Fetch URL with up to max_retries attempts.

    Returns (response_json, success) tuple.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.json().get("retry_after", 2)
                print(f"    Rate limited, waiting {retry_after}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_after)
                continue

            # Handle server errors
            if response.status_code in [500, 502, 503, 504]:
                print(f"    Server error {response.status_code}, retrying in 3s (attempt {attempt + 1}/{max_retries})")
                time.sleep(3)
                continue

            # Handle other errors
            if response.status_code != 200:
                print(f"    Error {response.status_code}: {response.text[:100]}")
                time.sleep(1)
                continue

            return response.json(), True

        except requests.exceptions.Timeout:
            print(f"    Timeout, retrying (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
            continue
        except requests.exceptions.RequestException as e:
            print(f"    Request error: {e}, retrying (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
            continue

    return None, False


def fetch_polymarket_orderbooks(token_id, token_name):
    """
    Fetch all Polymarket orderbook snapshots for a token.

    Returns raw snapshots as-is from the API (no preprocessing).
    """
    print(f"\nFetching Polymarket {token_name} token orderbooks...")
    print(f"  Time range: {unix_ms_to_et_string(START_TIME_MS)} to {unix_ms_to_et_string(END_TIME_MS)}")

    headers = get_headers()
    all_snapshots = []
    pagination_key = None
    page = 1

    while True:
        # Build URL with pagination and time range
        url = f"{POLYMARKET_URL}?limit=200&token_id={token_id}&start_time={START_TIME_MS}&end_time={END_TIME_MS}"
        if pagination_key:
            url += f"&pagination_key={pagination_key}"

        # Fetch with retries
        data, success = fetch_with_retries(url, headers)

        if not success or data is None:
            print(f"  Failed to fetch page {page} after {MAX_RETRIES} retries")
            break

        snapshots = data.get("snapshots", [])

        if not snapshots:
            break

        # Store raw snapshots as-is
        all_snapshots.extend(snapshots)

        # Progress update
        if page % 5 == 0:
            print(f"  Page {page}: {len(all_snapshots)} snapshots collected")

        # Check for more pages
        pagination = data.get("pagination", {})
        if not pagination.get("has_more", False):
            break

        pagination_key = pagination.get("pagination_key")
        if not pagination_key:
            break

        page += 1

        # Rate-limited delay between requests (50 req/s = 0.025s delay)
        time.sleep(REQUEST_DELAY)

    print(f"  Completed: {len(all_snapshots)} {token_name} snapshots")

    # Add human-readable timestamp fields
    for snap in all_snapshots:
        if "timestamp" in snap:
            snap["timestamp_et"] = unix_ms_to_et_string(snap["timestamp"])
        if "indexedAt" in snap:
            snap["indexedAt_et"] = unix_ms_to_et_string(snap["indexedAt"])

    return all_snapshots


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("Rick Rieder Fed Chair Nomination Market - Orderbook Fetcher")
    print("=" * 70)
    print(f"\nMarket: Will Trump nominate Rick Rieder as the next Fed chair?")
    print(f"\nTime range: {unix_ms_to_et_string(START_TIME_MS)} to {unix_ms_to_et_string(END_TIME_MS)}")
    print(f"Rate limit: ~{int(1/REQUEST_DELAY)} requests/second")
    print(f"\nTokens:")
    print(f"  YES: {POLYMARKET_YES_TOKEN[:30]}...")
    print(f"  NO:  {POLYMARKET_NO_TOKEN[:30]}...")

    # Fetch YES and NO orderbooks in parallel (independent requests)
    print("\nFetching YES and NO tokens in parallel...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_yes = executor.submit(fetch_polymarket_orderbooks, POLYMARKET_YES_TOKEN, "YES")
        future_no = executor.submit(fetch_polymarket_orderbooks, POLYMARKET_NO_TOKEN, "NO")

        yes_snapshots = future_yes.result()
        no_snapshots = future_no.result()

    # Find actual time range from data
    all_timestamps = []
    for snap in yes_snapshots:
        if "timestamp" in snap:
            all_timestamps.append(snap["timestamp"])
    for snap in no_snapshots:
        if "timestamp" in snap:
            all_timestamps.append(snap["timestamp"])

    actual_start = min(all_timestamps) if all_timestamps else None
    actual_end = max(all_timestamps) if all_timestamps else None

    # Build output
    output = {
        "market_info": {
            "market": "Will Trump nominate Rick Rieder as the next Fed chair?",
            "platform": "Polymarket",
            "yes_token_id": POLYMARKET_YES_TOKEN,
            "no_token_id": POLYMARKET_NO_TOKEN,
            "time_range": {
                "start_ms": actual_start,
                "end_ms": actual_end,
                "start_et": unix_ms_to_et_string(actual_start) if actual_start else None,
                "end_et": unix_ms_to_et_string(actual_end) if actual_end else None
            },
            "data_note": "Raw orderbook snapshots from Dome API, no preprocessing applied"
        },
        "yes_snapshots": yes_snapshots,
        "no_snapshots": no_snapshots,
        "total_yes_snapshots": len(yes_snapshots),
        "total_no_snapshots": len(no_snapshots)
    }

    # Save to file
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "rick_rieder_orderbooks.json")

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print(f"{'=' * 70}")
    print(f"YES snapshots: {len(yes_snapshots)}")
    print(f"NO snapshots:  {len(no_snapshots)}")
    if actual_start and actual_end:
        print(f"\nActual data range:")
        print(f"  Start: {unix_ms_to_et_string(actual_start)}")
        print(f"  End:   {unix_ms_to_et_string(actual_end)}")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
