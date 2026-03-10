"""
build_dataset.py – Build a large Polymarket historical price dataset.

Instead of keyword search, this exhaustively paginates through every market
on Gamma, filters by volume, fetches full price history from CLOB, and saves
as Parquet files (efficient for querying with pandas or DuckDB).

Outputs:
  data/markets.parquet  – market catalog (one row per token/outcome)
  data/prices.parquet   – all price history (timestamp, token_id, price)

Usage:
  python data/build_dataset.py
  python data/build_dataset.py --min-volume 50000 --max-markets 2000
  python data/build_dataset.py --update   # append new markets only

Query examples (after building):
  import duckdb
  duckdb.sql("SELECT * FROM 'data/prices.parquet' WHERE token_id='...'")
  duckdb.sql(
      "SELECT slug, AVG(price) FROM 'data/prices.parquet' "
      "JOIN 'data/markets.parquet' USING (token_id) GROUP BY slug"
  )
"""
import argparse
import json
import logging
import os
import time

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com"

MARKETS_PATH = "data/markets.parquet"
PRICES_PATH = "data/prices.parquet"


# ── Step 1: catalog all markets ───────────────────────────────────────────────

def fetch_all_markets(
    min_volume: float = 10_000, page_size: int = 100
) -> pd.DataFrame:
    """
    Paginate through every Gamma market and return a flat DataFrame of
    (market_id, slug, question, outcome, token_id, volume, start_date,
     end_date, closed, tags).
    """
    rows, offset = [], 0
    log.info(f"Fetching market catalog (min_volume={min_volume:,.0f})…")

    while True:
        try:
            r = requests.get(
                GAMMA,
                params={
                    "limit": page_size,
                    "offset": offset,
                    "order": "volumeClob",
                    "ascending": False,
                },
                timeout=20,
            )
            r.raise_for_status()
        except Exception as e:
            log.warning(f"Gamma page error at offset {offset}: {e}")
            time.sleep(5)
            continue

        page = r.json()
        if not page:
            break

        added = 0
        for m in page:
            vol = float(m.get("volume") or 0)
            if vol < min_volume:
                # Sorted by volume desc; below threshold means we're done.
                log.info(
                    f"Reached volume threshold at offset {offset}, stopping."
                )
                return _to_df(rows)

            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            for i, tid in enumerate(token_ids):
                outcome = (
                    outcomes[i] if i < len(outcomes) else f"outcome_{i}"
                )
                rows.append({
                    "market_id": m["id"],
                    "slug": m.get("slug", ""),
                    "question": m.get("question", ""),
                    "outcome": outcome,
                    "token_id": tid,
                    "token_index": i,
                    "volume": vol,
                    "start_date": m.get("startDate"),
                    "end_date": m.get("endDate"),
                    "closed": m.get("closed", False),
                    "tags": json.dumps(m.get("tags") or []),
                })
                added += 1

        log.info(f"  offset {offset}: +{added} tokens ({len(rows)} total)")
        offset += page_size

        if len(page) < page_size:
            break  # last page

        time.sleep(0.2)

    return _to_df(rows)


def _to_df(rows):
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Step 2: fetch price history ───────────────────────────────────────────────

def fetch_history(
    token_id: str, interval: str = "max", retries: int = 3
) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{CLOB}/prices-history",
                params={
                    "market": token_id,
                    "interval": interval,
                    "fidelity": 60,
                },
                timeout=20,
            )
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning(f"Rate limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            history = r.json().get("history", [])
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history).rename(
                columns={"t": "timestamp", "p": "price"}
            )
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="s", utc=True
            )
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["token_id"] = token_id
            return df[["timestamp", "token_id", "price"]]
        except Exception as e:
            log.warning(f"History error attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def fetch_all_prices(
    markets_df: pd.DataFrame,
    yes_only: bool = True,
    interval: str = "max",
    sleep_between: float = 0.25,
    existing_token_ids: set = None,
) -> pd.DataFrame:
    """Fetch price history for every token in markets_df."""
    tokens = (
        markets_df[markets_df["token_index"] == 0] if yes_only else markets_df
    )

    if existing_token_ids:
        tokens = tokens[~tokens["token_id"].isin(existing_token_ids)]
        log.info(
            f"Skipping {len(existing_token_ids)} already-fetched tokens"
        )

    log.info(f"Fetching history for {len(tokens)} tokens…")
    frames = []

    for i, (_, row) in enumerate(tokens.iterrows(), 1):
        tid = row["token_id"]
        log.info(f"[{i}/{len(tokens)}] {row['slug']} / {row['outcome']}")
        df = fetch_history(tid, interval=interval)
        if df.empty:
            log.warning("  → no data")
            continue
        log.info(
            f"  → {len(df)} rows  "
            f"({df['timestamp'].min().date()} – "
            f"{df['timestamp'].max().date()})"
        )
        frames.append(df)
        time.sleep(sleep_between)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Step 3: save / update ─────────────────────────────────────────────────────

def load_existing_parquet(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    log.info(f"Saved {len(df):,} rows → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Polymarket historical dataset"
    )
    parser.add_argument(
        "--min-volume", type=float, default=1_000,
        help="Min lifetime volume USD (default: 10000)",
    )
    parser.add_argument(
        "--max-markets", type=int, default=None,
        help="Cap on number of markets to process (default: all)",
    )
    parser.add_argument(
        "--interval", default="max",
        choices=["1m", "1h", "1d", "1w", "max"],
    )
    parser.add_argument(
        "--all-outcomes", action="store_true",
        help="Fetch all outcomes, not just first/Yes",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Skip tokens already in prices.parquet",
    )
    args = parser.parse_args()

    # 1. Market catalog
    markets_df = fetch_all_markets(min_volume=args.min_volume)
    if markets_df.empty:
        log.error("No markets found.")
        return

    if args.max_markets:
        questions = markets_df["question"].unique()[:args.max_markets]
        markets_df = markets_df[markets_df["question"].isin(questions)]

    log.info(
        f"Catalog: {markets_df['question'].nunique():,} markets, "
        f"{len(markets_df):,} tokens"
    )

    # Save/merge market catalog
    existing_markets = load_existing_parquet(MARKETS_PATH)
    if not existing_markets.empty:
        combined = pd.concat(
            [existing_markets, markets_df]
        ).drop_duplicates("token_id")
    else:
        combined = markets_df
    save_parquet(combined, MARKETS_PATH)

    # 2. Price history
    existing_prices = load_existing_parquet(PRICES_PATH)
    existing_tids = (
        set(existing_prices["token_id"].unique())
        if not existing_prices.empty and args.update
        else None
    )

    prices_df = fetch_all_prices(
        markets_df,
        yes_only=not args.all_outcomes,
        interval=args.interval,
        existing_token_ids=existing_tids,
    )

    if prices_df.empty:
        log.warning("No new price data.")
        return

    # Merge with existing
    if not existing_prices.empty:
        prices_df = pd.concat(
            [existing_prices, prices_df]
        ).drop_duplicates(["timestamp", "token_id"])

    save_parquet(prices_df, PRICES_PATH)

    # Summary
    print(f"\n{'=' * 60}")
    print(
        f"Markets catalog : "
        f"{combined['question'].nunique():,} unique markets"
    )
    print(f"Price rows      : {len(prices_df):,}")
    print(f"Tokens covered  : {prices_df['token_id'].nunique():,}")
    print(
        f"Date range      : "
        f"{prices_df['timestamp'].min()} – "
        f"{prices_df['timestamp'].max()}"
    )
    print("\nFiles:")
    print(f"  {MARKETS_PATH}")
    print(f"  {PRICES_PATH}")
    print("\nQuery example:")
    print("  import duckdb")
    print("  duckdb.sql(")
    print("      \"SELECT m.slug, p.timestamp, p.price\"")
    print("      \"FROM 'data/prices.parquet' p\"")
    print("      \"JOIN 'data/markets.parquet' m USING (token_id)\"")
    print("      \"WHERE m.slug = 'some-market-slug'\"")
    print("  )")


if __name__ == "__main__":
    main()
