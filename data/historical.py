import requests
import json
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL  = "https://clob.polymarket.com"


def search_markets(
    keywords: List[str],
    limit: int = 10,
    closed: Optional[bool] = None,
) -> List[Dict]:
    """Return markets matching any of the given keywords, sorted by volume."""
    seen, results = set(), []

    for kw in keywords:
        params = {"limit": limit, "title": kw, "order": "volume", "ascending": False}
        if closed is not None:
            params["closed"] = str(closed).lower()

        r = requests.get(GAMMA_URL, params=params, timeout=10)
        if r.status_code != 200:
            logger.warning(f"Gamma API error for '{kw}': {r.status_code}")
            continue

        for m in r.json():
            if m["id"] in seen:
                continue
            seen.add(m["id"])

            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            for i, token_id in enumerate(token_ids):
                outcome = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
                results.append({
                    "market_id":    m["id"],
                    "slug":         m["slug"],
                    "question":     m["question"],
                    "outcome":      outcome,
                    "token_id":     token_id,
                    "condition_id": m["conditionId"],
                    "volume":       float(m.get("volume") or 0),
                    "start_date":   m.get("startDate"),
                    "end_date":     m.get("endDate"),
                    "closed":       m.get("closed", False),
                })

    results.sort(key=lambda x: x["volume"], reverse=True)
    return results


def fetch_history(token_id: str, interval: str = "max") -> pd.DataFrame:
    """
    Fetch full price history for a single token.

    interval options: '1m' (all ticks), '1h', '1d', '1w', 'max'
    Returns a DataFrame with columns: timestamp, price
    """
    r = requests.get(
        f"{CLOB_URL}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": 60},
        timeout=10,
    )
    if r.status_code != 200:
        logger.error(f"prices-history error for {token_id}: {r.status_code}")
        return pd.DataFrame()

    history = r.json().get("history", [])
    if not history:
        logger.warning(f"No history returned for {token_id}")
        return pd.DataFrame()

    df = pd.DataFrame(history).rename(columns={"t": "timestamp", "p": "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def fetch_histories(markets: List[Dict], interval: str = "max") -> pd.DataFrame:
    """
    Fetch history for a list of market dicts (as returned by search_markets).
    Returns a combined DataFrame with market metadata columns attached.
    """
    frames = []
    for m in markets:
        logger.info(f"Fetching history: {m['slug']} / {m['outcome']}")
        df = fetch_history(m["token_id"], interval=interval)
        if df.empty:
            continue
        for col in ("slug", "question", "outcome", "token_id", "volume"):
            df[col] = m[col]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    cols = ["timestamp", "slug", "question", "outcome", "token_id", "price", "volume"]
    return combined[[c for c in cols if c in combined.columns]]


def save(df: pd.DataFrame, prefix: str = "history") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{prefix}_{ts}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    keywords = ["russia", "ukraine", "israel", "iran", "china", "taiwan", "ceasefire"]

    print("Searching markets…")
    markets = search_markets(keywords, limit=10, closed=None)
    print(f"Found {len(markets)} market/outcome pairs. Top 10 by volume:")
    for m in markets[:10]:
        status = "closed" if m["closed"] else "open"
        print(f"  [{status}] {m['question']} — {m['outcome']}  (${m['volume']/1e6:.2f}M)")

    # Keep only the first token (Yes) of each question to avoid duplication
    seen_q, top_markets = set(), []
    for m in markets:
        key = m["question"]
        if key not in seen_q:
            seen_q.add(key)
            top_markets.append(m)
        if len(top_markets) >= 5:
            break

    print(f"\nFetching full history for {len(top_markets)} markets…")
    df = fetch_histories(top_markets, interval="max")

    if df.empty:
        print("No data retrieved.")
    else:
        print(df.groupby(["slug", "outcome"])[["price"]].describe().round(3))
        path = save(df, prefix="data/geopolitical_history")
        print(f"\nSaved to {path}")
