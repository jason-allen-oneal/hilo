import json
from dataclasses import dataclass
from pathlib import Path

from lib.polymarket.models import Basket


@dataclass(frozen=True)
class BotConfig:
    baskets: tuple[Basket, ...]


def load_config(path: Path) -> BotConfig:
    payload = json.loads(path.read_text())
    baskets = []
    for basket in payload.get("baskets", []):
        token_ids = tuple(str(token_id) for token_id in basket.get("token_ids", []))
        baskets.append(
            Basket(
                name=basket["name"],
                description=basket.get("description"),
                token_ids=token_ids,
            )
        )
    return BotConfig(baskets=tuple(baskets))
