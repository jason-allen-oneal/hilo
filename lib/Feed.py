# lib/Feed.py

from abc import ABC, abstractmethod
from typing import List
from lib.types import Candle

class Feed(ABC):
    def __init__(self, symbol: str, interval: str = "1m"):
        self.symbol = symbol
        self.interval = interval

    @abstractmethod
    def fetch_latest(self, limit: int = 1) -> List[Candle]:
        """
        Fetch the most recent candles.
        Must return candles in ascending time order.
        """
        raise NotImplementedError