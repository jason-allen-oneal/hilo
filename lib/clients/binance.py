# lib/clients/binance.py

import requests
from typing import List
from lib.types import Candle
from lib.Feed import Feed

class BinanceFeed(Feed):
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def fetch_latest(self, limit: int = 1) -> List[Candle]:
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit,
        }
        r = requests.get(self.BASE_URL, params=params, timeout=5)
        r.raise_for_status()

        candles = []
        for k in r.json():
            candles.append(
                Candle(
                    exchange="binance",
                    symbol=self.symbol,
                    interval=self.interval,
                    open_time=k[0],
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    close_time=k[6],
                )
            )
        return candles