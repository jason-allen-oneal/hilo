# lib/clients/coinbase.py

import requests
from typing import List
from lib.types import Candle
from lib.Feed import Feed

class CoinbaseFeed(Feed):
    BASE_URL = "https://api.exchange.coinbase.com/products"

    def fetch_latest(self, limit: int = 1) -> List[Candle]:
        granularity = 60  # seconds for 1m

        url = f"{self.BASE_URL}/{self.symbol}/candles"
        params = {
            "granularity": granularity,
            "limit": limit,
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()

        # Coinbase returns candles newest-first
        raw = sorted(r.json(), key=lambda x: x[0])

        candles = []
        for c in raw:
            open_time_sec = c[0]
            candles.append(
                Candle(
                    exchange="coinbase",
                    symbol=self.symbol,
                    interval=self.interval,
                    open_time=open_time_sec * 1000,
                    close_time=(open_time_sec + granularity) * 1000,
                    low=float(c[1]),
                    high=float(c[2]),
                    open=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                )
            )
        return candles