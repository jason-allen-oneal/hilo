import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from lib.polymarket.models import OrderBook, Quote


@dataclass(frozen=True)
class PolymarketClient:
    base_url: str
    timeout_s: int = 10

    def fetch_book(self, token_id: str) -> OrderBook:
        response = requests.get(
            f"{self.base_url}/book",
            params={"token_id": token_id},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return _parse_book(token_id, payload)


@dataclass(frozen=True)
class LocalBookSource:
    path: Path

    def load_book(self, token_id: str) -> OrderBook:
        payload = json.loads(self.path.read_text())
        if token_id not in payload:
            raise KeyError(f"Token {token_id} not found in {self.path}")
        return _parse_book(token_id, payload[token_id])


def _parse_book(token_id: str, payload: dict[str, Any]) -> OrderBook:
    bids = tuple(_parse_quotes(payload.get("bids", [])))
    asks = tuple(_parse_quotes(payload.get("asks", [])))
    return OrderBook(token_id=token_id, bids=bids, asks=asks)


def _parse_quotes(quotes: list[dict[str, Any]]) -> list[Quote]:
    parsed: list[Quote] = []
    for quote in quotes:
        price = float(quote["price"])
        size = float(quote.get("size", 0))
        parsed.append(Quote(price=price, size=size))
    return parsed
