from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QSizePolicy, QScrollArea, QTextEdit,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT
)


class _Canvas(FigureCanvasQTAgg):
    """Canvas that forwards wheel events to the parent scroll area."""
    def wheelEvent(self, event):
        event.ignore()  # let QScrollArea handle scrolling

# Available canvas area inside the result window (px).
# Toolbar ≈ 40 px, tab bar ≈ 35 px, margins ≈ 10 px.
_CANVAS_W = 1150
_CANVAS_H = 710


class ResultWindow(QWidget):
    """Independent window: report text as first tab, one figure tab each after."""

    def __init__(self, figures: list, title: str,
                 report_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1200, 800)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ── Tab 0: backtesting report text ────────────────────────────────
        if report_text:
            te = QTextEdit()
            te.setReadOnly(True)
            te.setPlainText(report_text)
            te.setFont(QFont("Courier New", 14))
            te.setStyleSheet(
                "QTextEdit { background:#111111; color:#d4d4d4; "
                "border:none; padding:20px; }"
            )
            te.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            tabs.addTab(te, "Report")

        # ── Figure tabs ───────────────────────────────────────────────────
        if not figures and not report_text:
            from PySide6.QtWidgets import QLabel
            layout.addWidget(QLabel("No output produced."))
            return

        for i, fig in enumerate(figures):
            fw, fh = fig.get_size_inches()
            # Render at 100 DPI; if the figure would exceed the canvas just fit it,
            # but never go below 72 DPI so text stays readable.
            ref_dpi = 100
            fit_scale = min(_CANVAS_W / (fw * ref_dpi),
                            _CANVAS_H / (fh * ref_dpi))
            dpi = max(72, min(ref_dpi, int(ref_dpi * fit_scale)))
            fig.set_dpi(dpi)
            suptitle = getattr(fig, '_suptitle', None)
            top_rect = 0.90 if suptitle is not None else 0.93
            try:
                fig.tight_layout(pad=1.5, rect=[0, 0, 1, top_rect])
            except Exception:
                try:
                    fig.subplots_adjust(top=top_rect, bottom=0.08, left=0.08, right=0.97)
                except Exception:
                    pass
            # Move suptitle baseline inside the figure so the text doesn't clip
            if suptitle is not None:
                suptitle.set_y(0.96)

            # Natural pixel size of the figure at this DPI
            px_w = int(fw * dpi)
            px_h = int(fh * dpi)

            canvas = _Canvas(fig)
            canvas.draw()   # render now so the canvas isn't black before the event loop starts
            # Fix canvas to its natural size so the scroll area can scroll it
            canvas.setFixedSize(px_w, px_h)

            toolbar = NavigationToolbar2QT(canvas, self)

            scroll = QScrollArea()
            scroll.setWidgetResizable(False)   # keep canvas at its natural size
            scroll.setWidget(canvas)
            scroll.setAlignment(Qt.AlignCenter)

            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(2)
            vbox.addWidget(toolbar)
            vbox.addWidget(scroll)

            tabs.addTab(container, f"Figure {i + 1}")

    def show_and_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()
