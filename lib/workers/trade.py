from PySide6.QtCore import QThread, Signal
import threading
from lib.utils import get_ticker, build_context
from lib.buffer import RollingCandleBuffer
from lib.rounds import RoundTracker, ROUND_MINUTES
from threading import Event
from typing import Optional, Callable, Dict, Tuple
from lib.model.model import RoundModel
from pathlib import Path

import time
from datetime import datetime, timezone, timedelta

def sleep_until_next_minute():
    now = datetime.now(timezone.utc)
    seconds = 60 - (now.second + now.microsecond / 1_000_000)
    time.sleep(max(seconds, 0))

def _infer(model: Optional[RoundModel], ctx: dict):
    decision = "HOLD"
    p_up = None
    confidence = 0.0
    reason = "PREDICTING"

    # XGBoost gives more confident predictions (less clustering near 0.5),
    # so we use a higher bias threshold to filter out weak signals.
    min_bias = 0.02

    if model is None:
        reason = "NO_MODEL"
        return decision, p_up, confidence, reason

    p_up = float(model.predict_proba(ctx))
    bias = p_up - 0.5
    confidence = abs(bias) * 2

    if abs(bias) < min_bias:
        decision = "HOLD"
    elif bias > 0:
        decision = "UP"
    else:
        decision = "DOWN"

    return decision, p_up, confidence, reason


def _prev_round_bounds(dt: datetime) -> Tuple[int, int]:
    """
    Given a timezone-aware datetime, return start/end timestamps (ms)
    of the immediately previous 15-minute round.
    """
    prev_dt = dt - timedelta(minutes=ROUND_MINUTES)
    prev_round_idx = prev_dt.minute // ROUND_MINUTES
    start_minute = prev_round_idx * ROUND_MINUTES
    start_dt = prev_dt.replace(minute=start_minute, second=0, microsecond=0)
    end_dt = start_dt + timedelta(minutes=ROUND_MINUTES, seconds=-1, microseconds=0)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


def _find_round_close(buffer: RollingCandleBuffer, start_ts: int, end_ts: int) -> Optional[float]:
    candles = buffer.candles_between(start_ts, end_ts + 1)
    if not candles:
        return None
    return candles[-1].close

def trading_loop(
    stop_event: Event,
    update_fn: Callable[[dict], None],
):
    # Keep exchange aligned with training data + live price feed
    exchange = "coinbase"
    symbol = "BTC-USD"
    interval = "1m"

    feed = get_ticker(exchange, symbol, interval)
    buffer = RollingCandleBuffer(capacity=60)
    rounds = RoundTracker(decision_window_minutes=1)

    locked_prediction: Optional[Dict] = None  # {"round_id": (hour, round_idx), "prediction": str}
    last_outcome: Optional[Dict] = None       # {"round_id": (hour, round_idx), "prediction": str, "actual": str, "correct": bool}
    accuracy_total = 0
    accuracy_correct = 0
    last_round_close: Optional[float] = None
    last_round_close_ts: Optional[int] = None

    model_path = Path(__file__).resolve().parent.parent / "model" / "model.joblib"
    model: Optional[RoundModel] = None
    if model_path.exists():
        try:
            model = RoundModel(str(model_path))
        except Exception as exc:
            print(f"[WARN] Failed to load model at {model_path}: {exc}")
            model = None
    else:
        print(f"[WARN] Model file not found at {model_path}")

    # Backfill
    backfilled = False
    while not backfilled and not stop_event.is_set():
        try:
            for c in feed.fetch_latest(limit=60):
                buffer.append(c)
            backfilled = True
        except Exception as exc:
            print(f"[WARN] Trading backfill failed for {exchange}:{symbol}: {exc}")
            time.sleep(3)
            continue

    # Initial fill from latest buffer candle
    latest = buffer.latest()
    if latest:
        ctx = build_context(buffer)
        if ctx:
            ri = rounds.update(latest.close_time)
            dt = datetime.fromtimestamp(latest.close_time / 1000, tz=timezone.utc)
            prev_start, prev_end = _prev_round_bounds(dt)
            last_round_close = _find_round_close(buffer, prev_start, prev_end)
            with open("change_log.txt", "a") as f:
                f.write(f"Initial last_round_close: {last_round_close}\n")
            decision, p_up, confidence, reason = _infer(model, ctx)
            change_pct = ((ctx["price_now"] / last_round_close) - 1.0) * 100.0 if last_round_close else None
            update_fn({
                "price": ctx["price_now"],
                "decision": decision,
                "p_up": p_up,
                "confidence": confidence,
                "reason": reason,
                "round": ri.round_index,
                "minute": ri.minute_in_round,
                "ret_1m": ctx["ret_1m"],
                "ret_5m": ctx["ret_5m"],
                "ret_15m": ctx["ret_15m"],
                "vol_15m": ctx["vol_15m"],
                "vol_60m": ctx["vol_60m"],
                "range_15m": ctx["range_15m"],
                "volume_15m": ctx["volume_15m"],
                "timestamp": ctx["timestamp"],
                "locked_prediction": locked_prediction["prediction"] if locked_prediction else None,
                "last_outcome": last_outcome,
                "accuracy_correct": accuracy_correct,
                "accuracy_total": accuracy_total,
                "accuracy_pct": (accuracy_correct / accuracy_total * 100.0) if accuracy_total else None,
                "last_round_close": last_round_close,
                "last_round_close_ts": last_round_close_ts,
            })

    last_consumed_close_time: Optional[int] = None

    while not stop_event.is_set():
        if stop_event.wait(30):
            break

        try:
            candles = feed.fetch_latest(limit=2)
            appended = False
            for c in candles:
                if buffer.append(c):
                    appended = True
            latest = buffer.latest()
            if not latest:
                continue

            if appended:
                if last_consumed_close_time and latest.close_time <= last_consumed_close_time:
                    continue
                last_consumed_close_time = latest.close_time

            ctx = build_context(buffer)
            if not ctx:
                continue

            ri = rounds.update(latest.close_time)
            decision, p_up, confidence, reason = _infer(model, ctx)

            round_id: Tuple[int, int] = (ri.hour, ri.round_index)

            # Capture previous round close when a new round starts
            if appended and ri.minute_in_round == 0 and ri.prev_round_start_ts and ri.prev_round_end_ts:
                prev_close_candidate = _find_round_close(
                    buffer,
                    ri.prev_round_start_ts,
                    ri.prev_round_end_ts,
                )
                if prev_close_candidate is not None:
                    last_round_close = prev_close_candidate
                    last_round_close_ts = ri.prev_round_end_ts
                    with open("change_log.txt", "a") as f:
                        f.write(f"Updated last_round_close: {last_round_close}\n")

            # Lock the prediction once we enter the final decision window.
            if appended and ri.is_decision_window:
                if locked_prediction is None or locked_prediction.get("round_id") != round_id:
                    locked_prediction = {
                        "round_id": round_id,
                        "prediction": decision,
                        "timestamp": latest.close_time,
                    }
                    with open("lock_log.txt", "a") as f:
                        f.write(f"Prediction locked: {locked_prediction}\n")

            # Score the previous round at round end
            if appended and ri.is_round_end and locked_prediction and locked_prediction.get("round_id") == round_id:
                prev_close: Optional[float] = None
                if ri.prev_round_start_ts and ri.prev_round_end_ts:
                    prev_candles = buffer.candles_between(
                        ri.prev_round_start_ts,
                        ri.prev_round_end_ts + 1,
                    )
                    if prev_candles:
                        prev_close = prev_candles[-1].close

                actual_direction = None
                if prev_close is not None:
                    actual_direction = "UP" if latest.close > prev_close else "DOWN"

                was_correct = (
                    actual_direction is not None
                    and locked_prediction["prediction"] in ("UP", "DOWN")
                    and locked_prediction["prediction"] == actual_direction
                )

                if locked_prediction["prediction"] in ("UP", "DOWN") and actual_direction is not None:
                    accuracy_total += 1
                    if was_correct:
                        accuracy_correct += 1

                last_outcome = {
                    "round_id": round_id,
                    "prediction": locked_prediction["prediction"],
                    "actual": actual_direction,
                    "correct": was_correct if actual_direction is not None else None,
                }
                locked_prediction = None

            update_fn({
                "price": ctx["price_now"],
                "decision": decision,
                "p_up": p_up,
                "confidence": confidence,
                "reason": reason,
                "round": ri.round_index,
                "minute": ri.minute_in_round,
                "ret_1m": ctx["ret_1m"],
                "ret_5m": ctx["ret_5m"],
                "ret_15m": ctx["ret_15m"],
                "vol_15m": ctx["vol_15m"],
                "vol_60m": ctx["vol_60m"],
                "range_15m": ctx["range_15m"],
                "volume_15m": ctx["volume_15m"],
                "timestamp": ctx["timestamp"],
                "locked_prediction": locked_prediction["prediction"] if locked_prediction else None,
                "last_outcome": last_outcome,
                "accuracy_correct": accuracy_correct,
                "accuracy_total": accuracy_total,
                "accuracy_pct": (accuracy_correct / accuracy_total * 100.0) if accuracy_total else None,
                "last_round_close": last_round_close,
                "last_round_close_ts": last_round_close_ts,
            })

        except Exception:
            continue

class TradingWorker(QThread):
    model_update = Signal(dict)

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()

    def run(self):
        trading_loop(
            stop_event=self.stop_event,
            update_fn=self.model_update.emit,
        )

    def stop(self):
        self.stop_event.set()
