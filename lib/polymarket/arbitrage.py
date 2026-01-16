from dataclasses import dataclass

from lib.polymarket.models import Basket, BasketQuote, OrderBook, Quote


@dataclass(frozen=True)
class ArbitrageResult:
    basket_quote: BasketQuote
    edge: float


def build_basket_quote(basket: Basket, books: dict[str, OrderBook]) -> BasketQuote:
    per_token: list[tuple[str, Quote]] = []
    total_cost = 0.0
    worst_fill_size = float("inf")

    for token_id in basket.iter_token_ids():
        book = books[token_id]
        best_ask = book.best_ask()
        if best_ask is None:
            raise ValueError(f"No asks for token {token_id}")
        per_token.append((token_id, best_ask))
        total_cost += best_ask.price
        worst_fill_size = min(worst_fill_size, best_ask.size)

    if worst_fill_size == float("inf"):
        worst_fill_size = 0.0

    return BasketQuote(
        basket=basket,
        total_cost=total_cost,
        worst_fill_size=worst_fill_size,
        per_token=tuple(per_token),
    )


def evaluate_arbitrage(basket_quote: BasketQuote, min_edge: float) -> ArbitrageResult | None:
    edge = 1.0 - basket_quote.total_cost
    if edge >= min_edge:
        return ArbitrageResult(basket_quote=basket_quote, edge=edge)
    return None
