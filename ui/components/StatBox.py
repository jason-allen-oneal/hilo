from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
)

class StatBox(QWidget):
    def __init__(self, title: str, initial_value: str = "—"):
        super().__init__()
        
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.value_label = QLabel(initial_value)
        self.value_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(self.value_label)

    def update_value(self, new_value: str):
        self.value_label.setText(new_value)