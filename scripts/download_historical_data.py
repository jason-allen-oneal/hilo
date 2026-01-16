#!/usr/bin/env python3
"""
Download historical 1-minute BTC-USD candlestick data from Coinbase Pro.

Usage:
    python scripts/download_historical_data.py --months 12
    python scripts/download_historical_data.py --start-date 2023-01-01 --end-date 2024-01-01
"""

import argparse
import csv
import time
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import List, Optional

import ccxt

# Constants
MINUTE_MS = 60000  # Milliseconds in a minute


def download_candles(
    exchange: ccxt.Exchange,
    symbol: str,
    start_ts: int,
    end_ts: int,
    limit: int = 300,
) -> List[List]:
    """
    Download OHLCV candles from exchange within date range.
    
    Returns list of candles: [[timestamp, open, high, low, close, volume], ...]
    """
    all_candles = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            # Fetch batch of candles
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe='1m',
                since=current_ts,
                limit=limit
            )
            
            if not candles:
                break
            
            # Filter to only candles before end_ts
            candles = [c for c in candles if c[0] < end_ts]
            
            if not candles:
                break
                
            all_candles.extend(candles)
            
            # Move to next batch
            current_ts = candles[-1][0] + MINUTE_MS  # Add 1 minute in ms
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
            # Progress indicator
            progress = (current_ts - start_ts) / (end_ts - start_ts) * 100
            print(f"Progress: {progress:.1f}% ({len(all_candles)} candles)", end='\r')
            
        except ccxt.NetworkError as e:
            print(f"\n[WARN] Network error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print(f"\n[ERROR] Exchange error: {e}")
            break
    
    print()  # New line after progress
    return all_candles


def save_to_csv(candles: List[List], output_path: Path) -> None:
    """
    Save candles to CSV in the format expected by the training pipeline.
    
    Expected columns: open_time, open, high, low, close, volume
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['open_time', 'open', 'high', 'low', 'close', 'volume'])
        
        for candle in candles:
            timestamp, open_price, high, low, close, volume = candle
            writer.writerow([
                int(timestamp),
                float(open_price),
                float(high),
                float(low),
                float(close),
                float(volume)
            ])


def validate_data(candles: List[List]) -> None:
    """Check for gaps or issues in the downloaded data."""
    if not candles:
        print("[ERROR] No candles downloaded!")
        return
    
    # Sort by timestamp
    candles.sort(key=lambda x: x[0])
    
    # Check for gaps (missing minutes)
    gaps = 0
    for i in range(1, len(candles)):
        expected_ts = candles[i-1][0] + MINUTE_MS  # Previous + 1 minute
        actual_ts = candles[i][0]
        
        if actual_ts != expected_ts:
            gap_minutes = (actual_ts - expected_ts) // MINUTE_MS
            if gap_minutes > 0:
                gaps += gap_minutes
    
    if gaps > 0:
        print(f"[WARN] Found {gaps} missing minutes in the data")
    else:
        print("[OK] No gaps detected in the data")
    
    # Summary
    start_date = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
    end_date = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
    print(f"[OK] Downloaded {len(candles)} candles")
    print(f"[OK] Date range: {start_date} to {end_date}")


def main():
    parser = argparse.ArgumentParser(
        description="Download historical 1-minute BTC-USD data from Coinbase Pro"
    )
    parser.add_argument(
        '--months',
        type=int,
        default=12,
        help='Number of months to download (counting back from today)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). Overrides --months if provided.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). Defaults to today.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/historical_1m.csv'),
        help='Output CSV file path'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default='coinbase',
        choices=['coinbase', 'binance'],
        help='Exchange to download from'
    )
    
    args = parser.parse_args()
    
    # Setup exchange
    if args.exchange == 'coinbase':
        exchange = ccxt.coinbasepro({'enableRateLimit': True})
        symbol = 'BTC/USD'
    else:  # binance
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol = 'BTC/USDT'
    
    # Determine date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        start_date = end_date - relativedelta(months=args.months)
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    print(f"[INFO] Downloading {symbol} from {args.exchange}")
    print(f"[INFO] Date range: {start_date.date()} to {end_date.date()}")
    print(f"[INFO] Estimated candles: ~{(end_ts - start_ts) // 60000:,}")
    
    # Download data
    candles = download_candles(exchange, symbol, start_ts, end_ts)
    
    # Validate
    validate_data(candles)
    
    # Save
    save_to_csv(candles, args.output)
    print(f"[OK] Saved to {args.output}")


if __name__ == '__main__':
    main()
