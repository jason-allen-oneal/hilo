import time
from collections import deque

from PySide6.QtWidgets import QWidget, QHBoxLayout
from PySide6.QtCore import QTimer

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
from datetime import datetime

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
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        self.plot_window_seconds = 5 * 60
        self.scale_window_seconds = 15 * 60
        self.prices = deque()
        self._scale_min = None
        self._scale_max = None
        self._round_tracker = RoundTracker()
        self._last_round_close: float | None = None
        self._prev_price: float | None = None
        self._round_end_times = []
        self._vlines = []

        self.line, = self.ax.plot([], [], color="#89b4fa", linewidth=2)
        self.latest_price_marker = self.ax.scatter([], [], s=50, color="#f38ba8", zorder=10)
        self.threshold_line = self.ax.axhline(
            y=0,
            color="#cdd6f4",
            linestyle="--",
            linewidth=1,
            alpha=0.4,
            label="Last 15m close",
        )

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
            if round_info.prev_round_end_ts:
                self._round_end_times.append(round_info.prev_round_end_ts / 1000)

        self.prices.append((now, price))

        cutoff_scale = now - self.scale_window_seconds
        while self.prices and self.prices[0][0] < cutoff_scale:
            self.prices.popleft()

        if not self.prices:
            return

        cutoff_plot = now - self.plot_window_seconds
        self._round_end_times = [ts for ts in self._round_end_times if ts >= cutoff_plot]

        plot_points = [(t, p) for t, p in self.prices if t >= cutoff_plot]
        if not plot_points:
            plot_points = list(self.prices)[-1:]

        trail_points = plot_points[:-1]
        latest_t, latest_p = plot_points[-1]

        trail_x = [datetime.fromtimestamp(t) for t, _ in trail_points]
        trail_y = [p for _, p in trail_points]

        for line in self._vlines:
            line.remove()
        self._vlines.clear()

        for ts in self._round_end_times:
            vline = self.ax.axvline(x=datetime.fromtimestamp(ts), color="#f9e2af", linestyle="--", linewidth=1, alpha=0.5)
            self._vlines.append(vline)

        self.line.set_data(trail_x, trail_y)
        self.latest_price_marker.set_offsets([mdates.date2num(datetime.fromtimestamp(latest_t)), latest_p])

        if self._last_round_close is None:
            # If we haven't crossed a boundary yet, best-effort infer last round close from history.
            self._last_round_close = self._infer_last_round_close(now, round_info)

        if self._last_round_close is not None:
            color = "#a6e3a1" if price >= self._last_round_close else "#f38ba8"
            self.line.set_color(color)
            self.latest_price_marker.set_color(color)
            self.threshold_line.set_ydata([self._last_round_close, self._last_round_close])
            self.threshold_line.set_visible(True)
        else:
            self.line.set_color("#89b4fa")
            self.latest_price_marker.set_color("#89b4fa")
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

        start_time = now - (0.75 * self.plot_window_seconds)
        end_time = now + (0.25 * self.plot_window_seconds)
        self.ax.set_xlim(datetime.fromtimestamp(start_time), datetime.fromtimestamp(end_time))

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
