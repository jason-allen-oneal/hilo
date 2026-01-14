import time
from collections import deque

from PySide6.QtWidgets import QWidget, QHBoxLayout
from PySide6.QtCore import QTimer

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from lib.rounds import RoundTracker
from lib.workers.price import PriceWorker


class BottomPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.figure = Figure(facecolor="#1f2430")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#1f2430")

        self.ax.tick_params(colors="#cdd6f4")
        for spine in self.ax.spines.values():
            spine.set_color("#45475a")

        self.ax.set_title("BTC-USD (Live)", color="#cdd6f4", fontsize=10)

        self.plot_window_seconds = 5 * 60
        self.scale_window_seconds = 15 * 60
        self.prices = deque()
        self._scale_min = None
        self._scale_max = None
        self._round_tracker = RoundTracker()
        self._last_round_close: float | None = None
        self._prev_price: float | None = None

        self.line, = self.ax.plot([], [], color="#89b4fa", linewidth=2)
        self.threshold_line = self.ax.axhline(
            y=0,
            color="#cdd6f4",
            linestyle="--",
            linewidth=1,
            alpha=0.4,
            label="Last 15m close",
        )

        self.ax.set_xlim(0, self.plot_window_seconds)
        self.ax.set_ylim(0, 1)

        self.price_worker = PriceWorker("BTC-USD")
        self.price_worker.start()

        self.timer = QTimer(self)
        self.timer.setInterval(200)  # 5 FPS is more than enough for charts
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start()

    def refresh_plot(self):
        price = self.price_worker.get_latest_price()
        if price is None:
            return

        now = time.time()
        now_ms = int(now * 1000)

        round_info = self._round_tracker.update(now_ms)
        if round_info.round_just_ended:
            # Capture the "final" price of the completed 15m round using the last known price.
            self._last_round_close = self._prev_price or price

        self.prices.append((now, price))

        cutoff_scale = now - self.scale_window_seconds
        while self.prices and self.prices[0][0] < cutoff_scale:
            self.prices.popleft()

        if not self.prices:
            return

        cutoff_plot = now - self.plot_window_seconds
        plot_points = [(t, p) for t, p in self.prices if t >= cutoff_plot]
        if not plot_points:
            plot_points = list(self.prices)[-1:]

        oldest = plot_points[0][0]
        x = [t - oldest for t, _ in plot_points]
        y = [p for _, p in plot_points]

        self.line.set_data(x, y)
        if self._last_round_close is None:
            # If we haven't crossed a boundary yet, best-effort infer last round close from history.
            self._last_round_close = self._infer_last_round_close(now, round_info)

        if self._last_round_close is not None:
            color = "#a6e3a1" if price >= self._last_round_close else "#f38ba8"
            self.line.set_color(color)
            self.threshold_line.set_ydata([self._last_round_close, self._last_round_close])
            self.threshold_line.set_visible(True)
        else:
            self.line.set_color("#89b4fa")
            self.threshold_line.set_visible(False)

        window_min = min(p for _, p in self.prices)
        window_max = max(p for _, p in self.prices)

        if (
            self._scale_min is None
            or self._scale_max is None
            or price < self._scale_min
            or price > self._scale_max
        ):
            self._scale_min = window_min
            self._scale_max = window_max

        pad = max((self._scale_max - self._scale_min) * 0.02, self._scale_max * 0.0005, 0.5)
        self.ax.set_ylim(self._scale_min - pad, self._scale_max + pad)

        self.ax.set_xlim(0, self.plot_window_seconds)

        self.canvas.draw_idle()
        self._prev_price = price

    def closeEvent(self, event):
        self.timer.stop()
        self.price_worker.stop()
        self.price_worker.wait()
        super().closeEvent(event)

    def _infer_last_round_close(self, now: float, round_info):
        """
        Try to infer the most recent 15m close from stored prices so the line can color
        without waiting for the next boundary.
        """
        if not self.prices:
            return None

        round_start = now - (round_info.minute_in_round * 60)
        last_round_end = round_start

        # Walk backwards to find the last price at or before the previous round end.
        for t, p in reversed(self.prices):
            if t <= last_round_end:
                return p

        # Fallback: use the oldest recorded price if none fall before the boundary.
        return self.prices[0][1]
