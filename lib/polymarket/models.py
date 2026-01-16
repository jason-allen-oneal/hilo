from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Quote:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    token_id: str
    bids: tuple[Quote, ...]
    asks: tuple[Quote, ...]

    def best_ask(self) -> Quote | None:
        return min(self.asks, key=lambda quote: quote.price, default=None)

    def best_bid(self) -> Quote | None:
        return max(self.bids, key=lambda quote: quote.price, default=None)


@dataclass(frozen=True)
class Basket:
    name: str
    token_ids: tuple[str, ...]
    description: str | None = None

    def iter_token_ids(self) -> Iterable[str]:
        return self.token_ids


@dataclass(frozen=True)
class BasketQuote:
    basket: Basket
    total_cost: float
    worst_fill_size: float
    per_token: tuple[tuple[str, Quote], ...]
