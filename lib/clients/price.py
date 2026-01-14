
import asyncio
import json
from typing import AsyncGenerator, Optional

import websockets


class PriceTracker:
    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, product_id: str = "BTC-USD"):
        self.product_id = product_id

    async def _subscribe(self, ws) -> None:
        """
        Send a subscribe message for the ticker channel.
        """
        msg = {
            "type": "subscribe",
            "product_ids": [self.product_id],
            "channels": ["ticker"],
        }
        await ws.send(json.dumps(msg))

    async def stream(self) -> AsyncGenerator[float, None]:
        """
        Async generator that yields live prices.

        Automatically reconnects on disconnects/errors with a short delay.
        """
        while True:
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=1,
                ) as ws:
                    await self._subscribe(ws)

                    async for raw in ws:
                        data = json.loads(raw)
                        if (
                            data.get("type") == "ticker"
                            and data.get("product_id") == self.product_id
                        ):
                            price: Optional[str] = data.get("price")
                            if price is None:
                                continue
                            yield float(price)
            except Exception as exc:
                # Log and retry after a brief pause to avoid tight reconnect loops.
                print(f"[WARN] Coinbase price stream error: {exc}")
                await asyncio.sleep(3)

