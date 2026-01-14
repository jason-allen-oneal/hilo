# lib/utils.py

from lib.clients.binance import BinanceFeed
from lib.clients.coinbase import CoinbaseFeed
from lib.Feed import Feed
import math
from datetime import datetime, timezone
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
    closes = [c.close for c in candles]

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

    def ema(values: list[float], period: int) -> float:
        if period <= 0:
            raise ValueError("period must be > 0")
        alpha = 2.0 / (period + 1.0)
        ema_value = values[0]
        for value in values[1:]:
            ema_value = alpha * value + (1 - alpha) * ema_value
        return ema_value

    def rsi(period: int) -> float:
        if period <= 0:
            raise ValueError("period must be > 0")
        gains = []
        losses = []
        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            if delta >= 0:
                gains.append(delta)
            else:
                losses.append(-delta)
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def sma(values: list[float]) -> float:
        return sum(values) / len(values)

    def bollinger_position(period: int, num_std: float = 2.0) -> float:
        window = closes[-period:]
        mean = sma(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance)
        upper = mean + num_std * std
        lower = mean - num_std * std
        band_width = upper - lower
        if band_width == 0:
            return 0.5
        return (price_now - lower) / band_width

    def momentum(period: int) -> float:
        return price_now - closes[-period - 1]

    def price_distance_from_sma(period: int) -> float:
        average = sma(closes[-period:])
        return (price_now / average) - 1.0

    def macd_signal(macd_values: list[float]) -> float:
        return ema(macd_values, 9)

    last_15 = candles[-15:]
    high_15 = max(c.high for c in last_15)
    low_15 = min(c.low for c in last_15)

    volume_15 = sum(c.volume for c in last_15)

    macd_series = [
        ema(closes[:i], 12) - ema(closes[:i], 26)
        for i in range(len(closes) - 26, len(closes) + 1)
    ]
    macd_value = macd_series[-1]
    macd_signal_value = macd_signal(macd_series)

    timestamp_dt = datetime.fromtimestamp(latest.close_time / 1000, tz=timezone.utc)

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
        "rsi_14": rsi(14),
        "macd": macd_value,
        "macd_signal": macd_signal_value,
        "macd_hist": macd_value - macd_signal_value,
        "bb_position": bollinger_position(20),
        "hour_of_day": float(timestamp_dt.hour),
        "day_of_week": float(timestamp_dt.weekday()),
        "momentum_5m": momentum(5),
        "momentum_15m": momentum(15),
        "momentum_30m": momentum(30),
        "dist_sma_5": price_distance_from_sma(5),
        "dist_sma_15": price_distance_from_sma(15),
        "dist_sma_30": price_distance_from_sma(30),
        "dist_sma_60": price_distance_from_sma(60),
    }

def get_ticker(exchange: str, symbol: str, interval: str = "1m") -> Feed:
    exchange = exchange.lower()

    if exchange == "binance":
        return BinanceFeed(symbol, interval)
    if exchange == "coinbase":
        return CoinbaseFeed(symbol, interval)

    raise ValueError(f"Unsupported exchange: {exchange}")
