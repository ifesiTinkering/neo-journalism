#!/usr/bin/env python3
"""
Find causal relationships between interesting markets and all Polymarket markets.
Uses Claude API to identify which markets affect or are affected by the interesting markets.
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
from anthropic import Anthropic
from pathlib import Path
import time
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Load environment variables
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
POLYMARKET_API_BASE = "https://gamma-api.polymarket.com"

# Output file for results
OUTPUT_FILE = Path(__file__).parent / "market_relationships.md"


def parse_interesting_markets(filepath: str) -> list[dict]:
    """Parse the interesting_markets.txt file into a list of market dicts."""
    markets = []
    current_category = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if this is a category header
            if line.endswith('markets') or line.endswith('events') or line.endswith('event') or line.endswith('market'):
                current_category = line
                continue

            # Parse market line: "Name, URL"
            if ', https://' in line:
                parts = line.split(', https://')
                if len(parts) == 2:
                    markets.append({
                        'name': parts[0],
                        'url': 'https://' + parts[1],
                        'category': current_category
                    })

    return markets


def fetch_all_polymarket_events(limit_per_request: int = 100) -> list[dict]:
    """
    Fetch ALL events from Polymarket API with full pagination.
    Filters to only events that end after today.
    """
    all_events = []
    offset = 0

    # Only fetch events ending after today
    today = datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching Polymarket events (ending after {today})...")

    while True:
        url = f"{POLYMARKET_API_BASE}/events"
        params = {
            'limit': limit_per_request,
            'offset': offset,
            'active': 'true',
            'end_date_min': today  # Only events ending after today
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            events = response.json()

            if not events:
                print(f"  Pagination complete at offset {offset}")
                break

            all_events.extend(events)

            if len(all_events) % 500 == 0 or len(events) < limit_per_request:
                print(f"  Fetched {len(all_events)} events so far...")

            if len(events) < limit_per_request:
                # Last page - fewer results than requested
                break

            offset += limit_per_request
            time.sleep(0.1)  # Be nice to the API

        except requests.RequestException as e:
            print(f"Error fetching events at offset {offset}: {e}")
            break

    print(f"Total events fetched: {len(all_events)}")
    return all_events


def extract_markets_from_events(events: list[dict], filter_old: bool = True) -> list[dict]:
    """
    Extract individual markets from events.
    Optionally filter out old/ended markets based on end_date.
    """
    markets = []
    skipped_old = 0

    # Cutoff date - only include markets ending after Jan 1, 2025
    cutoff_date = datetime(2025, 1, 1)

    for event in events:
        event_title = event.get('title', 'Unknown Event')
        event_description = event.get('description', '')

        # Check event end date if filtering
        if filter_old:
            end_date_str = event.get('endDate') or event.get('end_date')
            if end_date_str:
                try:
                    # Parse ISO format date
                    end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    if end_date.replace(tzinfo=None) < cutoff_date:
                        skipped_old += 1
                        continue
                except (ValueError, TypeError):
                    pass  # Keep market if date parsing fails

        # Some events have nested markets
        if 'markets' in event and event['markets']:
            for market in event['markets']:
                question = market.get('question', event_title)
                markets.append({
                    'event_title': event_title,
                    'event_description': event_description,
                    'market_question': question,
                    'market_id': market.get('id', ''),
                    'slug': event.get('slug', '')
                })
        else:
            # Event itself is the market
            question = event.get('question', event_title)
            markets.append({
                'event_title': event_title,
                'event_description': event_description,
                'market_question': question,
                'market_id': event.get('id', ''),
                'slug': event.get('slug', '')
            })

    if filter_old:
        print(f"Skipped {skipped_old} old markets (end date before 2025)")

    return markets


def find_relationships_with_claude(
    interesting_markets: list[dict],
    other_markets: list[dict],
    client: Anthropic,
    batch_size: int = 50
) -> list[dict]:
    """
    Use Claude to find causal relationships between interesting markets and other markets.
    Returns a list of relationship dicts.
    """
    all_relationships = []

    # Format interesting markets for the prompt
    interesting_str = "\n".join([
        f"- {m['name']}" for m in interesting_markets
    ])

    # Process other markets in batches
    total_batches = (len(other_markets) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(other_markets), batch_size):
        batch = other_markets[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        print(f"  Processing batch {batch_num}/{total_batches}...")

        # Format batch markets
        batch_str = "\n".join([
            f"- {m['market_question']}" for m in batch
        ])

        prompt = f"""You are analyzing prediction markets to find DIRECT CAUSAL relationships.

## INTERESTING MARKETS (our focus):
{interesting_str}

## OTHER MARKETS (to check for relationships):
{batch_str}

## TASK:
For each "other market", determine if it has a DIRECT CAUSAL relationship with any of the "interesting markets".

IMPORTANT RULES:
1. Only identify relationships where one market DIRECTLY CAUSES or AFFECTS the outcome of another
2. "No relationship" is a valid and EXPECTED answer for most markets - do NOT stretch to find connections
3. If the relationship is indirect, tenuous, or requires multiple steps of reasoning, it is NOT a valid relationship
4. One "other market" can relate to multiple "interesting markets" if genuinely applicable

Examples of VALID relationships:
- "Trump Fed Chair nomination" AFFECTS "Fed decision in March" (direct policy impact)
- "US Economic Recession" AFFECTS "Fed decision" (Fed responds to economic conditions)
- "Fed decision" AFFECTS "Bitcoin price" (monetary policy affects crypto)
- "Giannis traded to Celtics" AFFECTS "2026 NBA Champion" (star player movement affects championship odds)
- "Chiefs win Super Bowl" AFFECTS "NFL MVP" (winning team players often win MVP)

Examples of INVALID relationships (too indirect):
- "NBA MVP" affects "Presidential Election" (no direct causal link)
- "Oscar winner" affects "Fed decision" (no causal relationship)
- Random individual game spread affects championship (too indirect)

## OUTPUT FORMAT:
Return a JSON array. If no relationships exist, return an empty array [].
Each relationship should be:
{{"interesting_market": "exact name from list", "other_market": "exact name from list", "relationship": "affects" or "is affected by", "reason": "brief explanation"}}

Only return the JSON array, nothing else."""

        try:
            response = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse the response
            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            relationships = json.loads(response_text)

            if relationships:
                print(f"    Found {len(relationships)} relationships in this batch")
                all_relationships.extend(relationships)
            else:
                print(f"    No relationships found in this batch")

        except json.JSONDecodeError as e:
            print(f"    Error parsing Claude response: {e}")
            print(f"    Response was: {response_text[:500]}...")
        except Exception as e:
            print(f"    Error calling Claude API: {e}")

        # Rate limiting
        time.sleep(1)

    return all_relationships


def write_results_table(relationships: list[dict], output_file: Path):
    """Write relationships to a markdown table."""

    # Group by interesting market
    grouped = {}
    for rel in relationships:
        key = rel['interesting_market']
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(rel)

    # Build markdown content
    content = "# Market Relationships Analysis\n\n"
    content += "This table shows Polymarket markets that have direct causal relationships with our interesting markets.\n\n"
    content += "| Interesting Market | Related Market | Relationship | Reason |\n"
    content += "|-------------------|----------------|--------------|--------|\n"

    for interesting_market, rels in sorted(grouped.items()):
        for rel in rels:
            content += f"| {rel['interesting_market']} | {rel['other_market']} | {rel['relationship']} | {rel['reason']} |\n"

    content += f"\n\n*Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    content += f"*Total relationships found: {len(relationships)}*\n"

    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"\nResults written to: {output_file}")


def main():
    # Check for API key
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return

    # Configuration
    BATCH_SIZE = 50

    # Initialize Anthropic client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Parse interesting markets
    interesting_markets_file = Path(__file__).parent / "interesting_makets.txt"
    print(f"Loading interesting markets from: {interesting_markets_file}")
    interesting_markets = parse_interesting_markets(interesting_markets_file)
    print(f"Found {len(interesting_markets)} interesting markets\n")

    # Fetch ALL Polymarket events (no artificial limit)
    events = fetch_all_polymarket_events()

    # Extract markets from events (filter old ones)
    print("\nExtracting and filtering markets...")
    all_markets = extract_markets_from_events(events, filter_old=True)
    print(f"Extracted {len(all_markets)} current markets")

    # Filter out markets that are already in our interesting list (by name similarity)
    interesting_names = {m['name'].lower() for m in interesting_markets}
    other_markets = [
        m for m in all_markets
        if m['market_question'].lower() not in interesting_names
    ]
    print(f"Markets to analyze (excluding interesting markets): {len(other_markets)}")

    # Estimate API calls
    num_api_calls = (len(other_markets) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nEstimated API calls: {num_api_calls} (batch size: {BATCH_SIZE})")
    print("Using Claude Opus for higher quality reasoning\n")

    # Find relationships using Claude
    print("Analyzing relationships with Claude...")
    relationships = find_relationships_with_claude(
        interesting_markets,
        other_markets,
        client,
        batch_size=BATCH_SIZE
    )

    print(f"\nTotal relationships found: {len(relationships)}")

    # Write results
    if relationships:
        write_results_table(relationships, OUTPUT_FILE)
    else:
        print("No relationships found.")
        with open(OUTPUT_FILE, 'w') as f:
            f.write("# Market Relationships Analysis\n\n")
            f.write("No direct causal relationships were found between the interesting markets and other Polymarket markets.\n")
            f.write(f"\n*Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")


if __name__ == "__main__":
    main()
