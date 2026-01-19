#!/usr/bin/env python3
"""
Plot Market Order Simulation Results

Creates 5 clean graphs:
1-4: Difference in avg price (buy - sell) for $20, $100, $500, $1000 budgets
5: Spread (best_ask - best_bid) over time

All plots use 15-minute intervals.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_FILE = os.path.join(PARENT_DIR, 'market_order_simulation.csv')

# Highlight time: Jan 13, 2026 11:15 AM ET
HIGHLIGHT_TIME = datetime(2026, 1, 13, 11, 15, 0)

# Style settings for clean graphs
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def parse_timestamp(ts_str):
    """Parse timestamp string like 'Jan 11 2026 12:17:23 AM ET' to datetime."""
    # Remove ' ET' suffix and parse
    ts_clean = ts_str.replace(' ET', '')
    return datetime.strptime(ts_clean, '%b %d %Y %I:%M:%S %p')


def load_and_resample(csv_file, interval='15min'):
    """Load CSV and resample to specified interval."""
    df = pd.read_csv(csv_file)

    # Parse timestamps
    df['datetime'] = df['timestamp_et'].apply(parse_timestamp)
    df = df.set_index('datetime')

    # Select numeric columns
    numeric_cols = ['best_ask', 'best_bid', 'spread',
                    'buy_20', 'sell_20', 'diff_20',
                    'buy_100', 'sell_100', 'diff_100',
                    'buy_500', 'sell_500', 'diff_500',
                    'buy_1000', 'sell_1000', 'diff_1000']

    # Resample to 15-minute intervals, taking the mean
    df_resampled = df[numeric_cols].resample(interval).mean()

    # Drop any NaN rows (intervals with no data)
    df_resampled = df_resampled.dropna()

    return df_resampled


def plot_diff(ax, df, budget, color):
    """Plot the diff for a specific budget."""
    col = f'diff_{budget}'
    ax.plot(df.index, df[col], color=color, linewidth=1.2)
    ax.fill_between(df.index, 0, df[col], color=color, alpha=0.3)

    ax.set_title(f'Effective Spread: ${budget} Market Order', fontweight='bold')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Price Difference ($)')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Add highlight line
    ax.axvline(x=HIGHLIGHT_TIME, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(HIGHLIGHT_TIME, ax.get_ylim()[1] * 0.95, ' Jan 13 11:15 AM',
            color='red', fontsize=8, ha='left', va='top')


def plot_spread(ax, df, color):
    """Plot the bid-ask spread."""
    ax.plot(df.index, df['spread'], color=color, linewidth=1.2)
    ax.fill_between(df.index, 0, df['spread'], color=color, alpha=0.3)

    ax.set_title('Bid-Ask Spread (Best Ask - Best Bid)', fontweight='bold')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Spread ($)')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Add highlight line
    ax.axvline(x=HIGHLIGHT_TIME, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(HIGHLIGHT_TIME, ax.get_ylim()[1] * 0.95, ' Jan 13 11:15 AM',
            color='red', fontsize=8, ha='left', va='top')


def main():
    print("Loading and resampling data to 15-minute intervals...")
    df = load_and_resample(CSV_FILE, '15min')
    print(f"Loaded {len(df)} intervals")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Colors for each plot
    colors = {
        20: '#2ecc71',      # Green
        100: '#3498db',     # Blue
        500: '#9b59b6',     # Purple
        1000: '#e74c3c',    # Red
        'spread': '#f39c12' # Orange
    }

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(12, 16))
    fig.suptitle('Rick Rieder Market - Orderbook Simulation Results\n(15-minute intervals, Jan 11-15 2026)',
                 fontsize=14, fontweight='bold', y=0.995)

    # Plot each budget's diff
    budgets = [20, 100, 500, 1000]
    for i, budget in enumerate(budgets):
        plot_diff(axes[i], df, budget, colors[budget])

    # Plot spread
    plot_spread(axes[4], df, colors['spread'])

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.35)

    # Save figure
    output_file = os.path.join(SCRIPT_DIR, 'simulation_plots.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plots to: {output_file}")

    # Also save individual plots
    for i, budget in enumerate(budgets):
        fig_single, ax_single = plt.subplots(figsize=(10, 4))
        plot_diff(ax_single, df, budget, colors[budget])
        plt.tight_layout()
        plot_file = os.path.join(SCRIPT_DIR, f'plot_diff_{budget}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved: {plot_file}")

    # Save spread plot
    fig_spread, ax_spread = plt.subplots(figsize=(10, 4))
    plot_spread(ax_spread, df, colors['spread'])
    plt.tight_layout()
    spread_file = os.path.join(SCRIPT_DIR, 'plot_spread.png')
    plt.savefig(spread_file, dpi=150, bbox_inches='tight')
    plt.close(fig_spread)
    print(f"Saved: {spread_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
