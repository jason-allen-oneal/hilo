from PySide6.QtCore import QThread, Signal
import asyncio
import time
from lib.clients.price import PriceTracker


class PriceWorker(QThread):
    price_received = Signal(float)

    def __init__(self, product_id: str, emit_hz: float = 5.0):
        super().__init__()
        self.product_id = product_id
        self._running = True
        self._latest_price: float | None = None

        self._emit_interval = 1.0 / max(emit_hz, 0.1)
        self._last_emit = 0.0

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        tracker = PriceTracker(self.product_id)
        async for price in tracker.stream():
            if not self._running:
                break

            self._latest_price = price

            now = time.monotonic()
            if now - self._last_emit >= self._emit_interval:
                self._last_emit = now
                self.price_received.emit(price)

    def get_latest_price(self) -> float | None:
        return self._latest_price

    def stop(self):
        self._running = False
