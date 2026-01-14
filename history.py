# fetch_history.py

import csv
import time
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

import requests

# We will use Coinbase candles endpoint directly because it supports time ranges.
BASE_URL = "https://api.exchange.coinbase.com/products"


def iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_coinbase_candles(
    product: str,
    granularity_sec: int,
    start: datetime,
    end: datetime,
    timeout: int = 10,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Returns list of tuples:
      (open_time_ms, open, high, low, close, volume)

    Coinbase response format:
      [ time, low, high, open, close, volume ]
    time is UNIX seconds.
    """
    url = f"{BASE_URL}/{product}/candles"
    params = {
        "granularity": granularity_sec,
        "start": iso(start),
        "end": iso(end),
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = r.json()
    raw.sort(key=lambda x: x[0])  # ascending by time

    out = []
    for c in raw:
        t_sec = int(c[0])
        out.append((
            t_sec * 1000,
            float(c[3]),  # open
            float(c[2]),  # high
            float(c[1]),  # low
            float(c[4]),  # close
            float(c[5]),  # volume
        ))
    return out


def main():
    product = "BTC-USD"
    granularity_sec = 60

    # How much history you want:
    days = 14  # change this (start small, then increase)

    # Coinbase limits how many candles you can get per request.
    # We'll request in chunks of 300 minutes (~5 hours) to stay safe.
    chunk_minutes = 300

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)

    print(f"[INFO] Fetching {days} days of 1m candles for {product}")
    print(f"[INFO] Range: {start.isoformat()} -> {end.isoformat()}")

    rows = []
    cur_end = end

    while cur_end > start:
        cur_start = max(start, cur_end - timedelta(minutes=chunk_minutes))

        try:
            batch = fetch_coinbase_candles(
                product=product,
                granularity_sec=granularity_sec,
                start=cur_start,
                end=cur_end,
            )
            rows.extend(batch)
            print(f"[INFO] Got {len(batch)} candles: {cur_start.isoformat()} -> {cur_end.isoformat()}")
        except Exception as e:
            print(f"[WARN] Failed chunk {cur_start.isoformat()} -> {cur_end.isoformat()}: {e}")
            time.sleep(2)
            cur_end = cur_start
            continue

        # Move window backwards
        cur_end = cur_start

        # Be polite to the API
        time.sleep(0.25)

    # Dedup + sort
    rows = sorted({r[0]: r for r in rows}.values(), key=lambda x: x[0])

    out_path = "data/historical_1m.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} candles to {out_path}")


if __name__ == "__main__":
    main()
