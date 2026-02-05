#!/usr/bin/env python3
"""
Find causal relationships between interesting markets and all Polymarket markets.
Uses Claude API to identify which markets affect or are affected by the interesting markets.

V2 Changes:
- Better prompt to filter out trivial same-event relationships
- Includes market slug in output for lookup
- Filters placeholder markets before sending to Claude
"""

import os
import sys
import json
import requests
import re
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
OUTPUT_FILE = Path(__file__).parent / "market_relationships_v3.md"

# Placeholder patterns to filter out before sending to Claude
PLACEHOLDER_PATTERNS = re.compile(
    r'\bTeam [A-Z]{1,2}\b|'
    r'\bPerson [A-Z]{1,2}\b|'
    r'\bPlayer [A-Z]{1,2}\b|'
    r'\bD Senate\b|'
    r'\bR Senate\b|'
    r'\bD House\b|'
    r'\bR House\b|'
    r'\bAny Other Team\b|'
    r'\banother team\b',
    re.IGNORECASE
)


def parse_interesting_markets(filepath: str) -> list[dict]:
    """Parse the interesting_markets.txt file into a list of market dicts."""
    markets = []
    current_category = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.endswith('markets') or line.endswith('events') or line.endswith('event') or line.endswith('market'):
                current_category = line
                continue

            if ', https://' in line:
                parts = line.split(', https://')
                if len(parts) == 2:
                    # Extract slug from URL
                    url = 'https://' + parts[1]
                    slug = url.split('/')[-1] if '/' in url else ''
                    markets.append({
                        'name': parts[0],
                        'url': url,
                        'slug': slug,
                        'category': current_category
                    })

    return markets


def fetch_all_polymarket_events(limit_per_request: int = 100) -> list[dict]:
    """Fetch all events from Polymarket API with full pagination."""
    all_events = []
    offset = 0
    today = datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching Polymarket events (ending after {today})...")

    while True:
        url = f"{POLYMARKET_API_BASE}/events"
        params = {
            'limit': limit_per_request,
            'offset': offset,
            'active': 'true',
            'end_date_min': today
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
                break

            offset += limit_per_request
            time.sleep(0.1)

        except requests.RequestException as e:
            print(f"Error fetching events at offset {offset}: {e}")
            break

    print(f"Total events fetched: {len(all_events)}")
    return all_events


def extract_markets_from_events(events: list[dict]) -> list[dict]:
    """Extract individual markets from events, including event context."""
    markets = []
    skipped_placeholders = 0

    for event in events:
        event_title = event.get('title', 'Unknown Event')
        event_slug = event.get('slug', '')
        event_description = event.get('description', '')

        # Some events have nested markets
        if 'markets' in event and event['markets']:
            for market in event['markets']:
                question = market.get('question', event_title)

                # Skip placeholder markets
                if PLACEHOLDER_PATTERNS.search(question):
                    skipped_placeholders += 1
                    continue

                markets.append({
                    'event_title': event_title,
                    'event_slug': event_slug,
                    'event_description': event_description,
                    'market_question': question,
                    'market_slug': market.get('slug', event_slug),
                    'market_id': market.get('id', ''),
                })
        else:
            question = event.get('question', event_title)

            # Skip placeholder markets
            if PLACEHOLDER_PATTERNS.search(question):
                skipped_placeholders += 1
                continue

            markets.append({
                'event_title': event_title,
                'event_slug': event_slug,
                'event_description': event_description,
                'market_question': question,
                'market_slug': event_slug,
                'market_id': event.get('id', ''),
            })

    print(f"Skipped {skipped_placeholders} placeholder markets")
    return markets


def find_relationships_with_claude(
    interesting_markets: list[dict],
    other_markets: list[dict],
    client: Anthropic,
    batch_size: int = 50,
    output_file: Path = None
) -> list[dict]:
    """Use Claude to find causal relationships between interesting markets and other markets.
    Saves results incrementally every 10 batches to avoid losing progress.
    """
    all_relationships = []

    # Format interesting markets with their slugs
    interesting_str = "\n".join([
        f"- {m['name']} (slug: {m['slug']})" for m in interesting_markets
    ])

    # Process other markets in batches
    total_batches = (len(other_markets) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(other_markets), batch_size):
        batch = other_markets[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        print(f"  Processing batch {batch_num}/{total_batches}...")

        # Format batch markets with event context
        batch_str = "\n".join([
            f"- {m['market_question']} (event: {m['event_title']}, slug: {m['market_slug']})"
            for m in batch
        ])

        prompt = f"""You are analyzing prediction markets to find MEANINGFUL CROSS-EVENT CAUSAL relationships.

## INTERESTING MARKETS (our focus):
{interesting_str}

## OTHER MARKETS (to check for relationships):
{batch_str}

## TASK:
Find markets that have DIRECT CAUSAL relationships with the interesting markets.

## CRITICAL RULES:

1. **NO SAME-EVENT RELATIONSHIPS**: If an "other market" is a sub-market within the same event as an "interesting market", DO NOT include it. These are trivial.

   BAD EXAMPLE (same event - DO NOT INCLUDE):
   - "Will Italy win the 2026 FIFA World Cup?" affecting "2026 FIFA World Cup Winner"
   - This is trivial because "Will Italy win?" is literally a market WITHIN the "2026 FIFA World Cup Winner" event

   BAD EXAMPLE (same event - DO NOT INCLUDE):
   - "Will JD Vance win 2028 presidential election?" affecting "Presidential Election Winner 2028"
   - This is trivial because it's a sub-market of the same event

2. **YES TO CROSS-EVENT CAUSAL RELATIONSHIPS**: Look for markets from DIFFERENT events that causally affect each other.

   GOOD EXAMPLE (cross-event causal):
   - "Who will Trump nominate as Fed Chair?" AFFECTS "Fed decision in March"
   - These are different events, but the Fed Chair appointment directly influences rate decisions

   GOOD EXAMPLE (cross-event causal):
   - "Will star player X transfer to Italy's national team?" AFFECTS "2026 FIFA World Cup Winner"
   - External recruitment affecting team's championship chances

   GOOD EXAMPLE (cross-event causal):
   - "US Economic Recession in 2025?" AFFECTS "Fed decision in March"
   - Economic conditions drive Fed policy decisions

3. **"No relationship" is expected for most markets** - do NOT stretch to find connections

4. **Skip placeholder markets** - anything with "Team A", "Person B", generic labels

## OUTPUT FORMAT:
Return a JSON array. If no MEANINGFUL cross-event relationships exist, return an empty array [].

Each relationship must include the market_slug for lookup:
{{"interesting_market": "exact name", "other_market": "exact market question", "other_market_slug": "the slug", "relationship": "affects" or "is affected by", "reason": "brief explanation of the CROSS-EVENT causal link"}}

Only return the JSON array, nothing else."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            relationships = json.loads(response_text)

            if relationships:
                print(f"    Found {len(relationships)} cross-event relationships in this batch")
                all_relationships.extend(relationships)
            else:
                print(f"    No cross-event relationships found in this batch")

        except json.JSONDecodeError as e:
            print(f"    Error parsing Claude response: {e}")
            print(f"    Response was: {response_text[:500]}...")
        except Exception as e:
            print(f"    Error calling Claude API: {e}")
            # Save progress on API error (like running out of credits)
            if output_file and all_relationships:
                print(f"    Saving {len(all_relationships)} relationships found so far...")
                save_intermediate_results(all_relationships, output_file)

        # Save progress every 50 batches
        if output_file and batch_num % 50 == 0 and all_relationships:
            print(f"    [Checkpoint] Saving {len(all_relationships)} relationships...")
            save_intermediate_results(all_relationships, output_file)

        time.sleep(1)

    return all_relationships


def save_intermediate_results(relationships: list[dict], output_file: Path):
    """Save intermediate results to avoid losing progress."""
    grouped = {}
    for rel in relationships:
        key = rel['interesting_market']
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(rel)

    content = "# Market Relationships Analysis (V3 - Cross-Event Only) [IN PROGRESS]\n\n"
    content += "This table shows Polymarket markets that have **meaningful cross-event causal relationships** with our interesting markets.\n"
    content += "Trivial same-event relationships (sub-markets) have been filtered out.\n\n"
    content += "| Interesting Market | Related Market | Slug | Relationship | Reason |\n"
    content += "|-------------------|----------------|------|--------------|--------|\n"

    for interesting_market, rels in sorted(grouped.items()):
        for rel in rels:
            slug = rel.get('other_market_slug', 'N/A')
            polymarket_link = f"[{slug}](https://polymarket.com/event/{slug})" if slug != 'N/A' else slug
            content += f"| {rel['interesting_market']} | {rel['other_market']} | {polymarket_link} | {rel['relationship']} | {rel['reason']} |\n"

    content += f"\n\n*Intermediate save at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    content += f"*Relationships found so far: {len(relationships)}*\n"

    with open(output_file, 'w') as f:
        f.write(content)


def write_results_table(relationships: list[dict], output_file: Path):
    """Write relationships to a markdown table with slugs for lookup."""

    # Group by interesting market
    grouped = {}
    for rel in relationships:
        key = rel['interesting_market']
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(rel)

    # Build markdown content
    content = "# Market Relationships Analysis (V3 - Cross-Event Only)\n\n"
    content += "This table shows Polymarket markets that have **meaningful cross-event causal relationships** with our interesting markets.\n"
    content += "Trivial same-event relationships (sub-markets) have been filtered out.\n\n"
    content += "| Interesting Market | Related Market | Slug | Relationship | Reason |\n"
    content += "|-------------------|----------------|------|--------------|--------|\n"

    for interesting_market, rels in sorted(grouped.items()):
        for rel in rels:
            slug = rel.get('other_market_slug', 'N/A')
            polymarket_link = f"[{slug}](https://polymarket.com/event/{slug})" if slug != 'N/A' else slug
            content += f"| {rel['interesting_market']} | {rel['other_market']} | {polymarket_link} | {rel['relationship']} | {rel['reason']} |\n"

    content += f"\n\n*Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    content += f"*Total cross-event relationships found: {len(relationships)}*\n"

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"\nResults written to: {output_file}")


def main():
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return

    BATCH_SIZE = 50

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Parse interesting markets
    interesting_markets_file = Path(__file__).parent / "interesting_makets.txt"
    print(f"Loading interesting markets from: {interesting_markets_file}")
    interesting_markets = parse_interesting_markets(interesting_markets_file)
    print(f"Found {len(interesting_markets)} interesting markets\n")

    # Fetch all Polymarket events
    events = fetch_all_polymarket_events()

    # Extract markets from events
    print("\nExtracting and filtering markets...")
    all_markets = extract_markets_from_events(events)
    print(f"Extracted {len(all_markets)} markets (after placeholder filtering)")

    # Filter out markets that are already in our interesting list
    interesting_names = {m['name'].lower() for m in interesting_markets}
    interesting_slugs = {m['slug'].lower() for m in interesting_markets}

    other_markets = [
        m for m in all_markets
        if m['market_question'].lower() not in interesting_names
        and m['market_slug'].lower() not in interesting_slugs
    ]
    print(f"Markets to analyze (excluding interesting markets): {len(other_markets)}")

    # Estimate API calls
    num_api_calls = (len(other_markets) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nEstimated API calls: {num_api_calls} (batch size: {BATCH_SIZE})")
    print("Using Claude Sonnet for cross-event relationship detection (cost-efficient)\n")

    # Find relationships (with incremental saving)
    print("Analyzing relationships with Claude...")
    relationships = find_relationships_with_claude(
        interesting_markets,
        other_markets,
        client,
        batch_size=BATCH_SIZE,
        output_file=OUTPUT_FILE
    )

    print(f"\nTotal cross-event relationships found: {len(relationships)}")

    # Write results
    if relationships:
        write_results_table(relationships, OUTPUT_FILE)
    else:
        print("No cross-event relationships found.")
        with open(OUTPUT_FILE, 'w') as f:
            f.write("# Market Relationships Analysis (V3)\n\n")
            f.write("No meaningful cross-event causal relationships were found.\n")
            f.write(f"\n*Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")


if __name__ == "__main__":
    main()
