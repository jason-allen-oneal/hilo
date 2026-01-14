# lib/utils.py

from lib.clients.binance import BinanceFeed
from lib.clients.coinbase import CoinbaseFeed
from lib.Feed import Feed
import math
from typing import Optional, Dict

from lib.buffer import RollingCandleBuffer


def build_context(buffer: RollingCandleBuffer) -> Optional[Dict[str, float]]:
    """
    Build a compact market context dict from a rolling candle buffer.
    Returns None if insufficient data exists.
    """
    if len(buffer) < 60:
        return None

    candles = buffer.as_list()
    latest = candles[-1]

    price_now = latest.close
    prev_15m_close = candles[-16].close

    # ---- Returns ----
    def ret(n: int) -> float:
        return (price_now / candles[-n-1].close) - 1.0

    # ---- Volatility ----
    def volatility(n: int) -> float:
        rets = [
            (candles[i].close / candles[i - 1].close) - 1.0
            for i in range(len(candles) - n + 1, len(candles))
        ]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    last_15 = candles[-15:]
    high_15 = max(c.high for c in last_15)
    low_15 = min(c.low for c in last_15)

    volume_15 = sum(c.volume for c in last_15)

    return {
        "timestamp": latest.close_time,
        "price_now": price_now,
        "prev_15m_close": prev_15m_close,

        "ret_1m": ret(1),
        "ret_5m": ret(5),
        "ret_15m": ret(15),

        "vol_15m": volatility(15),
        "vol_60m": volatility(60),

        "range_15m": (high_15 - low_15) / price_now,
        "volume_15m": volume_15,
    }

def get_ticker(exchange: str, symbol: str, interval: str = "1m") -> Feed:
    exchange = exchange.lower()

    if exchange == "binance":
        return BinanceFeed(symbol, interval)
    if exchange == "coinbase":
        return CoinbaseFeed(symbol, interval)

    raise ValueError(f"Unsupported exchange: {exchange}")