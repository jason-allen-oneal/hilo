from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt

from ui.top_panel import TopPanel
from ui.bottom_panel import BottomPanel

class MainLayout(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.top_panel = TopPanel()
        layout.addWidget(self.top_panel, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.bottom_panel = BottomPanel()
        layout.addWidget(self.bottom_panel)
