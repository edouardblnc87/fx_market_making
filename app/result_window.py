from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QSizePolicy
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT
)


class ResultWindow(QWidget):
    """Independent window showing one tab per captured matplotlib figure."""

    def __init__(self, figures: list, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1100, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if not figures:
            from PySide6.QtWidgets import QLabel
            layout.addWidget(QLabel("No figures produced."))
            return

        tabs = QTabWidget()
        layout.addWidget(tabs)

        for i, fig in enumerate(figures):
            canvas = FigureCanvasQTAgg(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            toolbar = NavigationToolbar2QT(canvas, self)

            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(2)
            vbox.addWidget(toolbar)
            vbox.addWidget(canvas)

            tab_label = f"Figure {i + 1}"
            tabs.addTab(container, tab_label)

    def show_and_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()
