#!/usr/bin/env python3
"""
Unix Timestamp Converter

Convert a date/time to Unix timestamp (seconds or milliseconds).

Usage:
    python unix_timestamp.py 2025 11 21 8 22 0 --tz EST
    python unix_timestamp.py 2025 11 21 8 22 0 --tz EST --ms
    python unix_timestamp.py 2025 11 21 8 22 --tz UTC
"""

import argparse
from datetime import datetime, timezone, timedelta

# Timezone offsets from UTC (in hours)
TIMEZONE_OFFSETS = {
    'UTC': 0,
    'EST': -5,
    'EDT': -4,
    'CST': -6,
    'CDT': -5,
    'MST': -7,
    'MDT': -6,
    'PST': -8,
    'PDT': -7,
}

def to_unix_timestamp(year, month, day, hour, minute, second=0, tz='EST', milliseconds=False):
    """Convert datetime to Unix timestamp."""

    # Get timezone offset
    tz_upper = tz.upper()
    if tz_upper not in TIMEZONE_OFFSETS:
        raise ValueError(f"Unknown timezone: {tz}. Available: {list(TIMEZONE_OFFSETS.keys())}")

    offset_hours = TIMEZONE_OFFSETS[tz_upper]

    # Create datetime and convert to UTC
    local_dt = datetime(year, month, day, hour, minute, second)
    utc_dt = local_dt - timedelta(hours=offset_hours)

    # Calculate Unix timestamp
    unix_seconds = int((utc_dt - datetime(1970, 1, 1)).total_seconds())

    if milliseconds:
        return unix_seconds * 1000
    return unix_seconds


def main():
    parser = argparse.ArgumentParser(
        description='Convert date/time to Unix timestamp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unix_timestamp.py 2025 11 21 8 22 0 --tz EST
  python unix_timestamp.py 2025 11 21 8 22 0 --tz EST --ms
  python unix_timestamp.py 2025 11 19 14 0 0 --tz EST --ms

Key Fed Events (EST):
  Nov 17, 2025 3:35 PM  - Waller speech
  Nov 19, 2025 2:00 PM  - FOMC Minutes
  Nov 20, 2025 8:30 AM  - Jobs Report
  Nov 21, 2025 8:22 AM  - Williams speech
        """
    )

    parser.add_argument('year', type=int, help='Year (e.g., 2025)')
    parser.add_argument('month', type=int, help='Month (1-12)')
    parser.add_argument('day', type=int, help='Day (1-31)')
    parser.add_argument('hour', type=int, help='Hour (0-23, 24-hour format)')
    parser.add_argument('minute', type=int, help='Minute (0-59)')
    parser.add_argument('second', type=int, nargs='?', default=0, help='Second (0-59, default: 0)')
    parser.add_argument('--tz', type=str, default='EST',
                        help=f"Timezone (default: EST). Options: {list(TIMEZONE_OFFSETS.keys())}")
    parser.add_argument('--ms', action='store_true',
                        help='Output in milliseconds (default: seconds)')

    args = parser.parse_args()

    try:
        timestamp = to_unix_timestamp(
            args.year, args.month, args.day,
            args.hour, args.minute, args.second,
            tz=args.tz,
            milliseconds=args.ms
        )

        # Display results
        unit = "milliseconds" if args.ms else "seconds"
        print(f"\nInput: {args.year}-{args.month:02d}-{args.day:02d} {args.hour:02d}:{args.minute:02d}:{args.second:02d} {args.tz.upper()}")
        print(f"Unix timestamp ({unit}): {timestamp}")

        # Also show the other format for convenience
        if args.ms:
            print(f"Unix timestamp (seconds): {timestamp // 1000}")
        else:
            print(f"Unix timestamp (milliseconds): {timestamp * 1000}")

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
