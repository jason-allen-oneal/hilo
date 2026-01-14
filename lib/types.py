# lib/types.py

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Candle:
    exchange: Literal["binance", "coinbase"]
    symbol: str
    interval: str          # e.g. "1m"
    open_time: int         # unix ms
    close_time: int        # unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float