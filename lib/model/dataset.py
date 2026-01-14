# lib/model/dataset.py

import csv
from pathlib import Path
from typing import List, Dict

from lib.utils import build_context
from lib.buffer import RollingCandleBuffer
from lib.rounds import RoundTracker
from lib.types import Candle


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

INPUT_CSV = "data/historical_1m.csv" 
OUTPUT_CSV = "data/round_dataset.csv"

BUFFER_SIZE = 60
DECISION_WINDOW_MINUTES = 2


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_candles_from_csv(path: str) -> List[Candle]:
    """
    Expected CSV columns:
      open_time, open, high, low, close, volume
    Times in ms (UTC).
    """
    candles: List[Candle] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                Candle(
                    exchange="coinbase",
                    symbol="BTC-USD",
                    interval="1m",
                    open_time=int(row["open_time"]),
                    close_time=int(row["open_time"]) + 60_000,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )

    candles.sort(key=lambda c: c.open_time)
    return candles


# ---------------------------------------------------------------------
# Main dataset build
# ---------------------------------------------------------------------

def build_round_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    buffer_size: int = BUFFER_SIZE,
    decision_window_minutes: int = DECISION_WINDOW_MINUTES,
) -> Path:
    """
    Build a round-level dataset from historical candles.
    Returns the output path for convenience.
    """
    in_path = Path(input_csv)
    out_path = Path(output_csv)

    candles = load_candles_from_csv(str(in_path))

    buffer = RollingCandleBuffer(capacity=buffer_size)
    rounds = RoundTracker(decision_window_minutes=decision_window_minutes)

    pending_round: Dict | None = None
    rows = []

    for c in candles:
        if not buffer.append(c):
            continue

        ctx = build_context(buffer)
        if not ctx:
            continue

        round_info = rounds.update(c.close_time)

        # -------------------------------------------------------------
        # Capture features at decision window
        # -------------------------------------------------------------
        if round_info.is_decision_window and pending_round is None:
            pending_round = {
                "round_start_time": c.close_time,
                "round_open_price": c.close,
                "features": ctx.copy(),
            }

        # -------------------------------------------------------------
        # Resolve label at round end
        # -------------------------------------------------------------
        if round_info.is_round_end and pending_round is not None:
            open_price = pending_round["round_open_price"]
            close_price = c.close

            label = 1 if close_price > open_price else 0

            row = pending_round["features"]
            row["label"] = label

            rows.append(row)
            pending_round = None

    # -----------------------------------------------------------------
    # Write dataset
    # -----------------------------------------------------------------

    if not rows:
        raise RuntimeError("No dataset rows generated")

    fieldnames = list(rows[0].keys())

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    build_round_dataset(INPUT_CSV, OUTPUT_CSV)
