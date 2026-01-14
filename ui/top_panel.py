import time
from collections import deque
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from lib.workers.price import PriceWorker
from lib.workers.trade import TradingWorker
from lib.rounds import ROUND_MINUTES
from ui.components.StatBox import StatBox

class TopPanel(QWidget):
    def __init__(self):
        super().__init__()

        panel_layout = QVBoxLayout(self)

        # Anchor everything to the top, and keep the two rows close together
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(2)
        panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        top_row = QHBoxLayout()
        bottom_row = QHBoxLayout()

        top_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        bottom_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(16)

        panel_layout.addLayout(top_row)
        panel_layout.addLayout(bottom_row)

        self.round_label = QLabel("Round: —")
        self.round_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_row.addWidget(self.round_label)

        self.time_label = QLabel("Time: --:--:--")
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_row.addWidget(self.time_label)

        self.price_box = StatBox("Price", "—")
        bottom_row.addWidget(self.price_box)

        self.prediction_box = StatBox("Prediction", "—")
        bottom_row.addWidget(self.prediction_box)

        self.change_box = StatBox("Change", "—")
        bottom_row.addWidget(self.change_box)

        self.locked_box = StatBox("Locked (round)", "—")
        bottom_row.addWidget(self.locked_box)

        self.accuracy_box = StatBox("Accuracy", "—")
        bottom_row.addWidget(self.accuracy_box)

        self.price_worker = PriceWorker("BTC-USD")
        self.trading_worker = TradingWorker()

        self.price_worker.price_received.connect(self.update_price)
        self.trading_worker.model_update.connect(self.update_layout)

        self._window_seconds = 60
        self._price_history = deque()  # (timestamp, price)
        self._last_round_close: float | None = None

        self.time_timer = QTimer(self)
        self.time_timer.setInterval(1000)
        self.time_timer.timeout.connect(self.update_time_label)
        self.time_timer.start()

        self.price_worker.start()
        self.trading_worker.start()

    def update_time_label(self):
        self.time_label.setText(datetime.now().strftime("Time: %H:%M:%S"))

    def update_price(self, price: float):
        now = time.time()
        self._price_history.append((now, price))
        cutoff = now - self._window_seconds
        while self._price_history and self._price_history[0][0] < cutoff:
            self._price_history.popleft()

        if self._last_round_close:
            change_pct = ((price / self._last_round_close) - 1.0) * 100.0
            self.change_box.update_value(f"{change_pct:+.3f}%")

        self.price_box.update_value(f"${price:,.2f}")

    def update_layout(self, data: dict):
        decision = data["decision"]
        p_up = data.get("p_up", None)

        rnd = data.get("round", "—")

        if isinstance(rnd, int):
            start_minute = rnd * ROUND_MINUTES
            end_minute = start_minute + (ROUND_MINUTES - 1)
            self.round_label.setText(f"Round: {start_minute}-{end_minute}")
        else:
            self.round_label.setText(f"Round: {rnd}")
        last_round_close = data.get("last_round_close")
        if last_round_close:
            self._last_round_close = last_round_close

        locked_pred = data.get("locked_prediction", None)
        last_outcome = data.get("last_outcome", None)

        if locked_pred:
            self.locked_box.update_value(f"{locked_pred}")
        elif last_outcome and last_outcome.get("prediction"):
            self.locked_box.update_value(f"{last_outcome.get('prediction')}")
        else:
            self.locked_box.update_value("—")

        display_prediction = None
        if last_outcome and last_outcome.get("actual"):
            actual = last_outcome["actual"]
            pred = last_outcome.get("prediction")
            correct = last_outcome.get("correct")
            suffix = "✓" if correct else "✕" if correct is not None else ""
            display_prediction = f"{pred} ({actual}) {suffix}"
        else:
            display_prediction = "HOLD (no model)" if p_up is None else f"{decision}"

        self.prediction_box.update_value(display_prediction)

        acc_total = data.get("accuracy_total", 0)
        acc_correct = data.get("accuracy_correct", 0)
        acc_pct = data.get("accuracy_pct", None)
        if acc_total and acc_pct is not None:
            self.accuracy_box.update_value(f"{acc_pct:.1f}% ({acc_correct}/{acc_total})")
        else:
            self.accuracy_box.update_value("—")
