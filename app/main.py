"""
app/main.py — Entry point for the FX Market Making desktop application.

Run with:
    python3.12 main.py
"""
from __future__ import annotations

import sys
import pathlib

import matplotlib
matplotlib.use("Agg")   # must be set before any other matplotlib import

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

from .main_window import MainWindow


def _apply_dark_palette(app: QApplication) -> None:
    """Set a dark base palette so native widgets (scrollbars, tooltips) also dark."""
    palette = QPalette()
    dark   = QColor("#1e1e1e")
    mid    = QColor("#252526")
    light  = QColor("#3c3c3c")
    text   = QColor("#d4d4d4")
    accent = QColor("#569cd6")
    palette.setColor(QPalette.Window,          dark)
    palette.setColor(QPalette.WindowText,      text)
    palette.setColor(QPalette.Base,            mid)
    palette.setColor(QPalette.AlternateBase,   dark)
    palette.setColor(QPalette.ToolTipBase,     mid)
    palette.setColor(QPalette.ToolTipText,     text)
    palette.setColor(QPalette.Text,            text)
    palette.setColor(QPalette.Button,          light)
    palette.setColor(QPalette.ButtonText,      text)
    palette.setColor(QPalette.BrightText,      QColor("#ffffff"))
    palette.setColor(QPalette.Highlight,       accent)
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text,       QColor("#555555"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#555555"))
    app.setPalette(palette)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("FX Market Making")
    app.setOrganizationName("Dauphine")

    _apply_dark_palette(app)

    qss_path = pathlib.Path(__file__).parent / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())

    win = MainWindow()
    win.resize(700, 520)
    win.show()
    return app.exec()
