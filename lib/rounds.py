# lib/rounds.py

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple


ROUND_MINUTES = 15


@dataclass(frozen=True)
class RoundInfo:
    # Current round
    hour: int
    round_index: int              # 0..3
    round_start_minute: int
    round_end_minute: int
    minute_in_round: int

    # State flags
    is_round_start: bool
    is_round_end: bool
    is_decision_window: bool

    # Transition info
    round_just_ended: bool

    # Previous round metadata (valid only if round_just_ended == True)
    prev_round_index: Optional[int]
    prev_round_start_ts: Optional[int]
    prev_round_end_ts: Optional[int]


class RoundTracker:
    """
    Tracks exact 15-minute rounds aligned to UTC clock:
      00–14, 15–29, 30–44, 45–59

    Provides explicit round transition metadata so callers can
    deterministically evaluate completed rounds.
    """

    def __init__(self, decision_window_minutes: int = 2):
        if decision_window_minutes <= 0 or decision_window_minutes >= ROUND_MINUTES:
            raise ValueError("decision_window_minutes must be between 1 and 14")

        self.decision_window_minutes = decision_window_minutes
        self._last_round_id: Optional[Tuple[int, int]] = None  # (hour, round_index)

    def update(self, timestamp_ms: int) -> RoundInfo:
        """
        Given a candle close_time in ms, return round metadata.
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

        minute = dt.minute
        hour = dt.hour

        round_index = minute // ROUND_MINUTES
        round_start_minute = round_index * ROUND_MINUTES
        round_end_minute = round_start_minute + (ROUND_MINUTES - 1)
        minute_in_round = minute - round_start_minute

        is_round_start = minute_in_round == 0
        is_round_end = minute_in_round == (ROUND_MINUTES - 1)
        is_decision_window = (
            minute_in_round >= (ROUND_MINUTES - self.decision_window_minutes - 1)
        )

        current_round_id = (hour, round_index)

        round_just_ended = (
            self._last_round_id is not None
            and current_round_id != self._last_round_id
        )

        prev_round_index = None
        prev_round_start_ts = None
        prev_round_end_ts = None

        if round_just_ended and self._last_round_id is not None:
            prev_hour, prev_round_index = self._last_round_id

            prev_round_start_minute = prev_round_index * ROUND_MINUTES
            prev_round_end_minute = prev_round_start_minute + (ROUND_MINUTES - 1)

            prev_round_start_dt = datetime(
                year=dt.year,
                month=dt.month,
                day=dt.day,
                hour=prev_hour,
                minute=prev_round_start_minute,
                second=0,
                tzinfo=timezone.utc,
            )

            prev_round_end_dt = datetime(
                year=dt.year,
                month=dt.month,
                day=dt.day,
                hour=prev_hour,
                minute=prev_round_end_minute,
                second=59,
                tzinfo=timezone.utc,
            )

            prev_round_start_ts = int(prev_round_start_dt.timestamp() * 1000)
            prev_round_end_ts = int(prev_round_end_dt.timestamp() * 1000)

        # Update internal state AFTER computing transition
        self._last_round_id = current_round_id

        return RoundInfo(
            hour=hour,
            round_index=round_index,
            round_start_minute=round_start_minute,
            round_end_minute=round_end_minute,
            minute_in_round=minute_in_round,
            is_round_start=is_round_start,
            is_round_end=is_round_end,
            is_decision_window=is_decision_window,
            round_just_ended=round_just_ended,
            prev_round_index=prev_round_index,
            prev_round_start_ts=prev_round_start_ts,
            prev_round_end_ts=prev_round_end_ts,
        )
