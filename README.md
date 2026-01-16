# Polymarket Arbitrage Bot

A lightweight scanner that checks Polymarket order books for basket arbitrage opportunities. Configure mutually exclusive outcome baskets (e.g., election outcomes) and the bot will flag when the combined cost to buy one share of every outcome is below 1.0.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start (Offline Sample)

```bash
python main.py --data fixtures/sample_books.json
```

## Live Scan

```bash
python main.py --live --base-url https://clob.polymarket.com --min-edge 0.02
```

## Configuration

Edit `config/arbitrage_baskets.json` to define baskets of mutually exclusive outcomes:

```json
{
  "baskets": [
    {
      "name": "Example election outcome basket",
      "description": "Three mutually exclusive outcomes for a sample event.",
      "token_ids": ["1001", "1002", "1003"]
    }
  ]
}
```

Each basket lists token IDs that should collectively resolve to exactly one winner. The bot sums the best ask across the basket and reports an edge when the total cost is below `1.0 - min_edge`.

## Project Structure

```
.
├── config/
│   └── arbitrage_baskets.json     # Basket definitions
├── fixtures/
│   └── sample_books.json          # Sample order books for offline runs
├── lib/
│   └── polymarket/                # Polymarket client + arbitrage logic
├── main.py                        # CLI entry point
└── requirements.txt
```
