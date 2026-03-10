"""
fetch_bulk.py  –  Pull historical price series for many Polymarket events.

Produces two outputs in data/:
  bulk_long_<ts>.csv      – long format (timestamp, slug, outcome, price, …)
  bulk_wide_<ts>.csv      – wide format (timestamp × token_id), good for backtesting

Usage:
  python data/fetch_bulk.py
  python data/fetch_bulk.py --categories politics crypto sports --per-category 20 --min-volume 50000
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB  = "https://clob.polymarket.com"

# ── Category keyword sets ────────────────────────────────────────────────────

CATEGORIES: Dict[str, List[str]] = {
    "politics_us":    ["trump", "biden", "president", "senate", "house", "supreme court",
                       "election", "republican", "democrat", "congress"],
    "politics_world": ["macron", "uk election", "germany", "canada election", "modi",
                       "xi jinping", "nato", "un", "g7", "g20"],
    "geopolitical":   ["russia", "ukraine", "israel", "iran", "china", "taiwan",
                       "ceasefire", "war", "sanctions", "north korea"],
    "crypto":         ["bitcoin", "ethereum", "solana", "btc", "eth", "crypto",
                       "coinbase", "binance", "doge", "xrp"],
    "macro_finance":  ["fed rate", "interest rate", "inflation", "recession", "gdp",
                       "unemployment", "oil price", "gold", "sp500", "nasdaq"],
    "sports":         ["nfl", "nba", "mlb", "nhl", "super bowl", "world cup",
                       "champions league", "wimbledon", "ufc", "formula 1"],
    "tech_ai":        ["openai", "anthropic", "google ai", "microsoft", "apple",
                       "nvidia", "ipo", "acquisition", "elon musk", "spacex"],
    "entertainment":  ["oscar", "grammy", "emmy", "nfl draft", "taylor swift",
                       "box office", "netflix", "academy award"],
    "climate_science":["climate", "hurricane", "earthquake", "temperature record",
                       "nasa", "spacex launch", "nobel prize"],
}


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def search_category(keywords: List[str], limit: int = 20,
                    min_volume: float = 10_000,
                    closed: Optional[bool] = None) -> List[Dict]:
    seen, results = set(), []
    for kw in keywords:
        params = {"limit": limit, "title": kw, "order": "volume", "ascending": False}
        if closed is not None:
            params["closed"] = str(closed).lower()
        try:
            r = requests.get(GAMMA, params=params, timeout=15)
            r.raise_for_status()
        except Exception as e:
            log.warning(f"Gamma error for '{kw}': {e}")
            continue

        for m in r.json():
            if m["id"] in seen:
                continue
            vol = float(m.get("volume") or 0)
            if vol < min_volume:
                continue
            seen.add(m["id"])

            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            for i, tid in enumerate(token_ids):
                outcome = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
                results.append({
                    "market_id":    m["id"],
                    "slug":         m["slug"],
                    "question":     m["question"],
                    "outcome":      outcome,
                    "token_id":     tid,
                    "token_index":  i,          # 0 = first/primary outcome
                    "volume":       vol,
                    "start_date":   m.get("startDate"),
                    "end_date":     m.get("endDate"),
                    "closed":       m.get("closed", False),
                })
    results.sort(key=lambda x: x["volume"], reverse=True)
    return results


def fetch_history(token_id: str, interval: str = "max",
                  retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{CLOB}/prices-history",
                params={"market": token_id, "interval": interval, "fidelity": 60},
                timeout=15,
            )
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning(f"Rate limited; waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            history = r.json().get("history", [])
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history).rename(columns={"t": "timestamp", "p": "price"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return df
        except Exception as e:
            log.warning(f"History fetch error ({attempt+1}/{retries}) for {token_id}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


# ── Main collection logic ─────────────────────────────────────────────────────

def collect(
    categories: List[str],
    per_category: int = 15,
    min_volume: float = 25_000,
    yes_only: bool = True,
    interval: str = "max",
    sleep_between: float = 0.3,
) -> pd.DataFrame:
    """
    Collect price history across categories.
    yes_only: keep only the first token per question (typically "Yes") to avoid redundancy.
    """
    all_markets: List[Dict] = []
    seen_questions: set = set()

    for cat in categories:
        if cat not in CATEGORIES:
            log.warning(f"Unknown category '{cat}', skipping")
            continue
        log.info(f"Searching category: {cat}")
        markets = search_category(CATEGORIES[cat], limit=per_category,
                                  min_volume=min_volume)
        cat_added = 0
        for m in markets:
            if yes_only and m["outcome"] not in ("Yes", "outcome_0", "Over"):
                continue
            key = m["question"]
            if key in seen_questions:
                continue
            seen_questions.add(key)
            m["category"] = cat
            all_markets.append(m)
            cat_added += 1
            if cat_added >= per_category:
                break
        log.info(f"  → {cat_added} markets added (total so far: {len(all_markets)})")

    log.info(f"\nTotal markets to fetch: {len(all_markets)}")

    frames = []
    for i, m in enumerate(all_markets, 1):
        log.info(f"[{i}/{len(all_markets)}] {m['slug']} / {m['outcome']}")
        df = fetch_history(m["token_id"], interval=interval)
        if df.empty:
            log.warning("  → no data, skipping")
            continue
        df["category"]  = m["category"]
        df["slug"]       = m["slug"]
        df["question"]   = m["question"]
        df["outcome"]    = m["outcome"]
        df["token_id"]   = m["token_id"]
        df["volume"]     = m["volume"]
        df["closed"]     = m["closed"]
        df["end_date"]   = m["end_date"]
        frames.append(df)
        log.info(f"  → {len(df)} rows  ({df['timestamp'].min().date()} – {df['timestamp'].max().date()})")
        time.sleep(sleep_between)

    if not frames:
        log.error("No data collected.")
        return pd.DataFrame()

    cols = ["timestamp", "category", "slug", "question", "outcome",
            "token_id", "price", "volume", "closed", "end_date"]
    combined = pd.concat(frames, ignore_index=True)
    return combined[[c for c in cols if c in combined.columns]]


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long → wide: index=timestamp (hourly), columns=token_id.
    Resamples to 1h and forward-fills short gaps (max 6h).
    """
    wide = (
        long_df
        .set_index("timestamp")
        .groupby("token_id")["price"]
        .resample("1h")
        .last()
        .unstack("token_id")
    )
    wide = wide.ffill(limit=6)
    # Rename columns to slug_outcome for readability
    meta = (
        long_df[["token_id", "slug", "outcome"]]
        .drop_duplicates("token_id")
        .set_index("token_id")
    )
    wide.columns = [
        f"{meta.loc[tid, 'slug']}__{meta.loc[tid, 'outcome']}"
        if tid in meta.index else tid
        for tid in wide.columns
    ]
    return wide


def save(df: pd.DataFrame, prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"data/{prefix}_{ts}.csv"
    df.to_csv(path, index=isinstance(df.index, pd.DatetimeIndex))
    log.info(f"Saved {len(df)} rows → {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bulk Polymarket historical data collector")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES.keys()),
                        help="Categories to fetch (default: all)")
    parser.add_argument("--per-category", type=int, default=15,
                        help="Max markets per category (default: 15)")
    parser.add_argument("--min-volume", type=float, default=25_000,
                        help="Min lifetime volume in USD (default: 25000)")
    parser.add_argument("--interval", default="max",
                        choices=["1m", "1h", "1d", "1w", "max"],
                        help="Price history interval (default: max)")
    parser.add_argument("--all-outcomes", action="store_true",
                        help="Fetch all outcomes (not just Yes/first token)")
    parser.add_argument("--no-wide", action="store_true",
                        help="Skip saving wide-format CSV")
    args = parser.parse_args()

    long_df = collect(
        categories=args.categories,
        per_category=args.per_category,
        min_volume=args.min_volume,
        yes_only=not args.all_outcomes,
        interval=args.interval,
    )

    if long_df.empty:
        print("No data collected.")
        return

    # Summary
    n_markets = long_df["token_id"].nunique()
    n_rows    = len(long_df)
    cats      = long_df["category"].value_counts().to_dict()
    print(f"\n{'='*60}")
    print(f"Collected {n_rows:,} rows across {n_markets} markets")
    print(f"Date range: {long_df['timestamp'].min()} – {long_df['timestamp'].max()}")
    print("\nMarkets per category:")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:<20} {cnt:>4} rows  ({long_df[long_df.category==cat]['token_id'].nunique()} markets)")

    long_path = save(long_df, "bulk_long")

    if not args.no_wide:
        wide_df = to_wide(long_df)
        wide_path = save(wide_df, "bulk_wide")
        print(f"\nWide matrix: {wide_df.shape[0]} timestamps × {wide_df.shape[1]} markets")
        print(f"  {wide_path}")

    print(f"\nLong format: {long_path}")


if __name__ == "__main__":
    main()
