# main.py

import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
)
from ui.layout import MainLayout


class TradingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hilo Dashboard")

        layout = QHBoxLayout()
        self.setLayout(layout)

        layout.addWidget(MainLayout())


def main():
    app = QApplication(sys.argv)
    win = TradingWindow()
    win.resize(600, 420)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
