"""
fetch_intraday.py – Download 1-minute intraday data for Polymarket contracts.

Usage:
  # All non-crypto markets with min volume $50k
  python data/fetch_intraday.py --all-non-crypto --min-volume 50000

  # By market slug (fetches Yes token)
  python data/fetch_intraday.py --slugs some-market-slug another-slug

  # By token ID directly
  python data/fetch_intraday.py --tokens TOKEN_ID1 TOKEN_ID2

  # Custom output path
  python data/fetch_intraday.py --all-non-crypto --out data/my_intraday.parquet
"""
import argparse
import json
import logging
import os
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com/markets"

DEFAULT_OUT = "data/intraday_1m.parquet"

# Keywords that identify crypto markets (checked against slug + question)
CRYPTO_KEYWORDS = {
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "ripple",
    "dogecoin", "doge", "litecoin", "ltc", "cardano", "ada", "polkadot",
    "avalanche", "avax", "chainlink", "link", "uniswap", "defi", "nft",
    "altcoin", "altcoins", "crypto", "stablecoin", "stablecoins", "blockchain",
    "coinbase", "binance", "bybit", "kraken", "celsius", "ftx", "tether",
    "usdc", "usdt", "dai", "memecoin", "meme coin", "fartcoin", "pepe",
    "shiba", "shib", "inu", "web3", "satoshi", "halving", "mining",
    "metaverse", "nft", "opensea", "blur", "polymarket token", "matic",
    "polygon", "near", "aptos", "sui", "ton", "tron", "stellar", "xlm",
}


def is_crypto(market: dict) -> bool:
    text = (market.get("slug", "") + " " + market.get("question", "")).lower()
    return any(kw in text for kw in CRYPTO_KEYWORDS)


def fetch_all_non_crypto_markets(
    min_volume: float = 10_000,
    page_size: int = 100,
) -> list[dict]:
    """Page through all Gamma markets, return non-crypto ones above min_volume."""
    results, offset = [], 0
    log.info(f"Scanning all markets (min_volume=${min_volume:,.0f})…")

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
            log.warning(f"Gamma error at offset {offset}: {e}")
            time.sleep(5)
            continue

        page = r.json()
        if not page:
            break

        for m in page:
            vol = float(m.get("volumeClob") or m.get("volume") or 0)
            if vol < min_volume:
                log.info(f"Reached volume threshold at offset {offset}, stopping.")
                return results

            if is_crypto(m):
                continue

            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            if not token_ids:
                continue

            # Yes token only (index 0)
            results.append({
                "token_id": token_ids[0],
                "slug": m.get("slug", ""),
                "question": m.get("question", ""),
                "outcome": outcomes[0] if outcomes else "Yes",
                "volume": vol,
            })

        log.info(f"  offset {offset}: {len(results)} non-crypto markets so far")
        offset += page_size

        if len(page) < page_size:
            break

        time.sleep(0.2)

    return results


def resolve_slugs(slugs: list[str]) -> list[dict]:
    """Look up token IDs for the given market slugs via Gamma API."""
    results = []
    for slug in slugs:
        r = requests.get(GAMMA, params={"slug": slug}, timeout=10)
        r.raise_for_status()
        markets = r.json()
        if not markets:
            log.warning(f"No market found for slug '{slug}'")
            continue
        m = markets[0]
        token_ids = m.get("clobTokenIds", "[]")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        outcomes = m.get("outcomes", "[]")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not token_ids:
            log.warning(f"No token IDs for '{slug}'")
            continue
        results.append({
            "token_id": token_ids[0],
            "slug": slug,
            "question": m.get("question", ""),
            "outcome": outcomes[0] if outcomes else "Yes",
            "volume": float(m.get("volumeClob") or m.get("volume") or 0),
        })
        log.info(f"Resolved '{slug}' → token {token_ids[0][:16]}…")
    return results


def fetch_1m(token_id: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{CLOB}/prices-history",
                params={"market": token_id, "interval": "1m", "fidelity": 1},
                timeout=30,
            )
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning(f"Rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            history = r.json().get("history", [])
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history).rename(columns={"t": "timestamp", "p": "price"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["token_id"] = token_id
            return df[["timestamp", "token_id", "price"]]
        except Exception as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def fetch_contracts(contracts: list[dict], out_path: str, sleep: float = 0.3):
    frames = []
    n = len(contracts)
    log.info(f"Fetching 1m data for {n} contract(s)…")

    for i, c in enumerate(contracts, 1):
        log.info(f"[{i}/{n}] {c['slug']} / {c['outcome']}")
        df = fetch_1m(c["token_id"])
        if df.empty:
            log.warning("  → no data")
            continue
        df["slug"] = c["slug"]
        df["outcome"] = c["outcome"]
        log.info(f"  → {len(df):,} rows  ({df['timestamp'].min().date()} – {df['timestamp'].max().date()})")
        frames.append(df)

        # Incremental save every 50 markets
        if i % 50 == 0:
            _merge_save(frames, out_path)
            frames = []

        time.sleep(sleep)

    if frames:
        _merge_save(frames, out_path)

    # Final summary
    if os.path.exists(out_path):
        result = pd.read_parquet(out_path)
        print(f"\n{'=' * 55}")
        print(f"Rows     : {len(result):,}")
        print(f"Tokens   : {result['token_id'].nunique()}")
        print(f"Markets  : {result['slug'].nunique()}")
        print(f"Range    : {result['timestamp'].min()} – {result['timestamp'].max()}")
        print(f"Output   : {out_path}")


def _merge_save(frames: list, path: str):
    if not frames:
        return
    new = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        new = pd.concat([existing, new]).drop_duplicates(["timestamp", "token_id"])
    new.to_parquet(path, index=False)
    log.info(f"  checkpoint: {len(new):,} total rows → {path}")


def main():
    parser = argparse.ArgumentParser(description="Download 1m intraday data for Polymarket contracts")
    parser.add_argument("--all-non-crypto", action="store_true",
                        help="Fetch all non-crypto markets above min-volume")
    parser.add_argument("--min-volume", type=float, default=10_000,
                        help="Min lifetime CLOB volume USD for --all-non-crypto (default: 10000)")
    parser.add_argument("--tokens", nargs="+", metavar="TOKEN_ID", default=[],
                        help="CLOB token IDs to fetch")
    parser.add_argument("--slugs", nargs="+", metavar="SLUG", default=[],
                        help="Market slugs to resolve and fetch (Yes token)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output parquet path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not args.all_non_crypto and not args.tokens and not args.slugs:
        parser.error("Provide --all-non-crypto, --tokens, or --slugs")

    contracts = []

    if args.all_non_crypto:
        contracts = fetch_all_non_crypto_markets(min_volume=args.min_volume)
        log.info(f"Found {len(contracts)} non-crypto markets")

    if args.tokens:
        contracts += [{"token_id": t, "slug": t[:12], "question": "", "outcome": "?"} for t in args.tokens]

    if args.slugs:
        contracts += resolve_slugs(args.slugs)

    if not contracts:
        log.error("No contracts to fetch.")
        return

    fetch_contracts(contracts, args.out)


if __name__ == "__main__":
    main()
