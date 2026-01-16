import argparse
from pathlib import Path

from lib.polymarket.arbitrage import build_basket_quote, evaluate_arbitrage
from lib.polymarket.client import LocalBookSource, PolymarketClient
from lib.polymarket.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket arbitrage scanner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/arbitrage_baskets.json"),
        help="Path to basket configuration JSON",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("fixtures/sample_books.json"),
        help="Local order book JSON for offline runs",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live order books from Polymarket CLOB",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://clob.polymarket.com",
        help="Base URL for the Polymarket CLOB API",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.01,
        help="Minimum edge required to flag an arbitrage opportunity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.live:
        client = PolymarketClient(base_url=args.base_url)
        fetch_book = client.fetch_book
    else:
        local_source = LocalBookSource(path=args.data)
        fetch_book = local_source.load_book

    if not config.baskets:
        raise SystemExit("No baskets configured. Add baskets in config/arbitrage_baskets.json")

    results = []
    for basket in config.baskets:
        books = {token_id: fetch_book(token_id) for token_id in basket.token_ids}
        basket_quote = build_basket_quote(basket, books)
        result = evaluate_arbitrage(basket_quote, args.min_edge)
        if result:
            results.append(result)

    if not results:
        print("No arbitrage opportunities found.")
        return

    for result in sorted(results, key=lambda item: item.edge, reverse=True):
        basket = result.basket_quote.basket
        print(f"\nBasket: {basket.name}")
        if basket.description:
            print(f"  {basket.description}")
        print(f"  Total cost: {result.basket_quote.total_cost:.4f}")
        print(f"  Edge: {result.edge:.4f}")
        print(f"  Worst fill size: {result.basket_quote.worst_fill_size:.2f}")
        for token_id, quote in result.basket_quote.per_token:
            print(f"    Token {token_id}: ask {quote.price:.4f} (size {quote.size:.2f})")


if __name__ == "__main__":
    main()
