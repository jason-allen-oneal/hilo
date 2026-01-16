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
        if len(closes) < period + 1:
            return 50.0  # Return neutral RSI if insufficient data
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

    # ---- Volume-based features ----
    def volume_trend(period: int) -> float:
        """Volume trend - is volume increasing?"""
        if len(candles) < period * 2:
            return 0.0
        recent_vol = sum(c.volume for c in candles[-period:])
        previous_vol = sum(c.volume for c in candles[-period*2:-period])
        if previous_vol == 0:
            return 0.0
        return (recent_vol / previous_vol) - 1.0

    def vwap_distance() -> float:
        """Volume-weighted average price distance"""
        last_15 = candles[-15:]
        total_vol = sum(c.volume for c in last_15)
        if total_vol == 0:
            return 0.0
        vwap = sum(c.close * c.volume for c in last_15) / total_vol
        return (price_now / vwap) - 1.0

    # ---- Volatility features ----
    def atr(period: int = 14) -> float:
        """ATR (Average True Range) - normalized"""
        if len(candles) < period + 1:
            return 0.0
        true_ranges = []
        for i in range(-period, 0):
            high_low = candles[i].high - candles[i].low
            high_close = abs(candles[i].high - candles[i-1].close)
            low_close = abs(candles[i].low - candles[i-1].close)
            true_ranges.append(max(high_low, high_close, low_close))
        avg_tr = sum(true_ranges) / len(true_ranges)
        return avg_tr / price_now if price_now > 0 else 0.0

    def bb_width(period: int = 20, num_std: float = 2.0) -> float:
        """Bollinger Band width (volatility proxy)"""
        if len(closes) < period:
            return 0.0
        window = closes[-period:]
        mean = sma(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance)
        if mean == 0:
            return 0.0
        return (2 * num_std * std) / mean

    # ---- Price action features ----
    def distance_from_high(period: int) -> float:
        """Recent high distance"""
        if len(candles) < period:
            return 0.0
        period_high = max(c.high for c in candles[-period:])
        return (price_now / period_high) - 1.0

    def distance_from_low(period: int) -> float:
        """Recent low distance"""
        if len(candles) < period:
            return 0.0
        period_low = min(c.low for c in candles[-period:])
        return (price_now / period_low) - 1.0

    def price_acceleration() -> float:
        """Price acceleration (rate of change of velocity)"""
        if len(closes) < 11:
            return 0.0
        # Velocity over last 5 periods vs velocity over previous 5 periods
        vel_now = closes[-1] - closes[-6]
        vel_prev = closes[-6] - closes[-11]
        return vel_now - vel_prev

    # ---- Cross-timeframe features ----
    def ema_alignment() -> float:
        """EMA alignment (trend confirmation across timeframes)"""
        if len(closes) < 30:
            return 0.0
        ema5 = ema(closes, 5)
        ema15 = ema(closes, 15)
        ema30 = ema(closes, 30)
        # Score: +1 if all aligned bullish, -1 if all bearish, 0 otherwise
        if ema5 > ema15 > ema30:
            return 1.0
        elif ema5 < ema15 < ema30:
            return -1.0
        else:
            return 0.0

    def rsi_divergence() -> float:
        """Multi-timeframe RSI divergence"""
        if len(closes) < 31:
            return 0.0
        rsi_14 = rsi(14)
        rsi_30 = rsi(30)
        return rsi_14 - rsi_30

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
        
        # New volume-based features
        "volume_trend_5m": volume_trend(5),
        "volume_trend_15m": volume_trend(15),
        "vwap_distance": vwap_distance(),
        
        # New volatility features
        "atr_14": atr(14),
        "bb_width": bb_width(20),
        
        # New price action features
        "distance_from_high_15m": distance_from_high(15),
        "distance_from_high_60m": distance_from_high(60),
        "distance_from_low_15m": distance_from_low(15),
        "distance_from_low_60m": distance_from_low(60),
        "price_acceleration": price_acceleration(),
        
        # New cross-timeframe features
        "ema_alignment": ema_alignment(),
        "rsi_divergence": rsi_divergence(),
    }

def get_ticker(exchange: str, symbol: str, interval: str = "1m") -> Feed:
    exchange = exchange.lower()

    if exchange == "binance":
        return BinanceFeed(symbol, interval)
    if exchange == "coinbase":
        return CoinbaseFeed(symbol, interval)

    raise ValueError(f"Unsupported exchange: {exchange}")
