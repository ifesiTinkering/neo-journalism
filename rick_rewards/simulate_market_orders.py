#!/usr/bin/env python3
"""
Market Order Simulation for Rick Rieder Orderbooks

Simulates market orders at various budget levels ($20, $100, $500, $1000)
and calculates average execution prices for both buying and selling.
"""

import json
import csv

# Budget levels to simulate
BUDGETS = [20, 100, 500, 1000]


def simulate_buy(asks, budget):
    """
    Simulate a market buy order for a given budget.

    Args:
        asks: List of asks from orderbook (will be sorted lowest price first)
        budget: Dollar amount to spend

    Returns:
        (shares_bought, avg_price) or (0, None) if can't fill
    """
    # Sort asks ascending (lowest price = best for buyer)
    sorted_asks = sorted(asks, key=lambda x: float(x['price']))

    spent = 0.0
    shares_bought = 0.0

    for ask in sorted_asks:
        price = float(ask['price'])
        size = float(ask['size'])
        level_cost = price * size

        remaining = budget - spent

        if remaining <= 0:
            break

        if level_cost <= remaining:
            # Take the whole level
            spent += level_cost
            shares_bought += size
        else:
            # Take partial level
            shares_at_this_level = remaining / price
            spent += remaining
            shares_bought += shares_at_this_level
            break

    if shares_bought == 0:
        return 0, None

    avg_price = spent / shares_bought
    return shares_bought, avg_price


def simulate_sell(bids, target_proceeds):
    """
    Simulate a market sell order to receive target proceeds.

    Args:
        bids: List of bids from orderbook (will be sorted highest price first)
        target_proceeds: Dollar amount to receive

    Returns:
        (shares_sold, avg_price) or (0, None) if can't fill
    """
    # Sort bids descending (highest price = best for seller)
    sorted_bids = sorted(bids, key=lambda x: float(x['price']), reverse=True)

    received = 0.0
    shares_sold = 0.0

    for bid in sorted_bids:
        price = float(bid['price'])
        size = float(bid['size'])
        level_revenue = price * size

        remaining = target_proceeds - received

        if remaining <= 0:
            break

        if level_revenue <= remaining:
            # Take the whole level
            received += level_revenue
            shares_sold += size
        else:
            # Take partial level
            shares_at_this_level = remaining / price
            received += remaining
            shares_sold += shares_at_this_level
            break

    if shares_sold == 0:
        return 0, None

    avg_price = received / shares_sold
    return shares_sold, avg_price


def process_snapshot(snapshot, budgets):
    """
    Process a single snapshot and return simulation results for all budgets.

    Returns dict with timestamp and results for each budget.
    """
    asks = snapshot.get('asks', [])
    bids = snapshot.get('bids', [])

    # Best ask = lowest ask price
    best_ask = min(float(a['price']) for a in asks) if asks else None

    # Best bid = highest bid price
    best_bid = max(float(b['price']) for b in bids) if bids else None

    # Spread = best_ask - best_bid
    spread = (best_ask - best_bid) if (best_ask is not None and best_bid is not None) else None

    result = {
        'timestamp': snapshot.get('timestamp'),
        'timestamp_et': snapshot.get('timestamp_et'),
        'best_ask': best_ask,
        'best_bid': best_bid,
        'spread': spread,
    }

    for budget in budgets:
        # Simulate buy
        _, buy_avg = simulate_buy(asks, budget)

        # Simulate sell
        _, sell_avg = simulate_sell(bids, budget)

        # Calculate difference
        if buy_avg is not None and sell_avg is not None:
            diff = buy_avg - sell_avg
        else:
            diff = None

        result[f'buy_{budget}'] = buy_avg
        result[f'sell_{budget}'] = sell_avg
        result[f'diff_{budget}'] = diff

    return result


def main():
    print("=" * 70)
    print("Market Order Simulation")
    print("=" * 70)

    # Load orderbook data
    with open('rick_rieder_orderbooks.json', 'r') as f:
        data = json.load(f)

    yes_snapshots = data['yes_snapshots']
    print(f"Loaded {len(yes_snapshots)} YES snapshots")
    print(f"Budgets: ${BUDGETS}")
    print()

    # Process all snapshots
    results = []
    for i, snap in enumerate(yes_snapshots):
        result = process_snapshot(snap, BUDGETS)
        results.append(result)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} snapshots...")

    print(f"Processed {len(results)} snapshots total")

    # Build header
    header = ['timestamp_et', 'best_ask', 'best_bid', 'spread']
    for budget in BUDGETS:
        header.extend([f'buy_{budget}', f'sell_{budget}', f'diff_{budget}'])

    # Save to CSV
    output_file = 'market_order_simulation.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for r in results:
            best_ask = r['best_ask']
            best_bid = r['best_bid']
            spread = r['spread']
            row = [
                r['timestamp_et'],
                f"{best_ask:.6f}" if best_ask is not None else "",
                f"{best_bid:.6f}" if best_bid is not None else "",
                f"{spread:.6f}" if spread is not None else "",
            ]
            for budget in BUDGETS:
                buy_val = r[f'buy_{budget}']
                sell_val = r[f'sell_{budget}']
                diff_val = r[f'diff_{budget}']

                row.append(f"{buy_val:.6f}" if buy_val is not None else "")
                row.append(f"{sell_val:.6f}" if sell_val is not None else "")
                row.append(f"{diff_val:.6f}" if diff_val is not None else "")

            writer.writerow(row)

    print(f"\nResults saved to: {output_file}")

    # Print sample of results
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS (first 5 snapshots)")
    print("=" * 70)

    for r in results[:5]:
        best_ask = r['best_ask']
        best_bid = r['best_bid']
        spread = r['spread']
        print(f"\n{r['timestamp_et']}")
        print(f"  Best Ask: ${best_ask:.4f}, Best Bid: ${best_bid:.4f}, Spread: ${spread:.4f}")
        for budget in BUDGETS:
            buy_val = r[f'buy_{budget}']
            sell_val = r[f'sell_{budget}']
            diff_val = r[f'diff_{budget}']

            buy_str = f"${buy_val:.4f}" if buy_val else "N/A"
            sell_str = f"${sell_val:.4f}" if sell_val else "N/A"
            diff_str = f"${diff_val:.4f}" if diff_val else "N/A"

            print(f"  ${budget:4d}: buy={buy_str}, sell={sell_str}, diff={diff_str}")


if __name__ == "__main__":
    main()
