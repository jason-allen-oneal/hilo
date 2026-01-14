from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple, Iterable

from lib.types import Candle


@dataclass(frozen=True)
class BufferStats:
    size: int
    capacity: int
    oldest_open_time: Optional[int]
    newest_open_time: Optional[int]


class RollingCandleBuffer:
    """
    Fixed-size rolling buffer of Candle objects.

    Guarantees:
      - Stores at most `capacity` candles
      - Rejects duplicates (same exchange+symbol+interval+open_time)
      - Rejects out-of-order candles (open_time <= last open_time)
      - Maintains strict time ordering
      - Supports time-window slicing for round evaluation
    """

    def __init__(self, capacity: int = 60):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.capacity = capacity
        self._buf: Deque[Candle] = deque(maxlen=capacity)
        self._seen_keys: set[Tuple[str, str, str, int]] = set()

    def _key(self, c: Candle) -> Tuple[str, str, str, int]:
        return (c.exchange, c.symbol, c.interval, c.open_time)

    def append(self, candle: Candle) -> bool:
        """
        Appends a candle if it is new and strictly in-order.

        Returns:
          True if appended
          False if rejected (duplicate or out-of-order)
        """
        k = self._key(candle)

        # Duplicate rejection
        if k in self._seen_keys:
            return False

        # Strict ordering
        if self._buf and candle.open_time <= self._buf[-1].open_time:
            return False

        # Evict oldest if full
        if len(self._buf) == self.capacity:
            oldest = self._buf[0]
            self._seen_keys.discard(self._key(oldest))

        self._buf.append(candle)
        self._seen_keys.add(k)
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def latest(self) -> Optional[Candle]:
        return self._buf[-1] if self._buf else None

    def oldest(self) -> Optional[Candle]:
        return self._buf[0] if self._buf else None

    def last(self, n: int) -> List[Candle]:
        """
        Returns the last n candles (or fewer if buffer is smaller).
        """
        if n <= 0:
            return []
        if n >= len(self._buf):
            return list(self._buf)
        return list(self._buf)[-n:]

    def as_list(self) -> List[Candle]:
        return list(self._buf)

    def __iter__(self) -> Iterable[Candle]:
        return iter(self._buf)

    # ------------------------------------------------------------------
    # Time-window helpers (NEW)
    # ------------------------------------------------------------------

    def candles_between(
        self,
        start_open_time: int,
        end_open_time: int,
    ) -> List[Candle]:
        """
        Returns candles whose open_time satisfies:
            start_open_time <= open_time < end_open_time

        This is the canonical method for round evaluation.
        """
        return [
            c for c in self._buf
            if start_open_time <= c.open_time < end_open_time
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._buf.clear()
        self._seen_keys.clear()

    def stats(self) -> BufferStats:
        if not self._buf:
            return BufferStats(
                size=0,
                capacity=self.capacity,
                oldest_open_time=None,
                newest_open_time=None,
            )

        return BufferStats(
            size=len(self._buf),
            capacity=self.capacity,
            oldest_open_time=self._buf[0].open_time,
            newest_open_time=self._buf[-1].open_time,
        )

    def __len__(self) -> int:
        return len(self._buf)
