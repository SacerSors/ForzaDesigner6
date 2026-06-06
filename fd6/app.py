import sys
from pathlib import Path

# IMPORTANT: Load PyTorch BEFORE Qt/Mesa on the main thread.
# Lazy-loading heavy PyTorch modules (like libtriton) from a worker/GUI thread
# on Linux/ROCm can cause fatal static TLS exhaustion and LLVM symbol collisions.
try:
    import torch
except ImportError:
    pass

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from fd6.gui.main_window import MainWindow
from fd6.gui.splash import maybe_show_splash


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Forza Designer 6")
    app.setOrganizationName("FD6")
    # Load bundled TTFs and apply the user's saved font choice (or the
    # "Default" pseudo-family = Segoe UI Variable → Segoe UI → sans fallback).
    from fd6.gui.fonts import load_bundled_fonts, apply_font, saved_font_name
    load_bundled_fonts()
    apply_font(app, saved_font_name())
    # Window icon matches the current theme's badge (Default → Pink)
    from fd6.gui.brand_banner import badge_path
    from fd6.gui.themes import badge_filename_for_theme, saved_theme_name
    bp = badge_path(badge_filename_for_theme(saved_theme_name()))
    if bp:
        app.setWindowIcon(QIcon(str(bp)))

    # Apply persisted theme before constructing MainWindow so styling applies cleanly
    from fd6.gui.themes import apply_theme, saved_theme_name
    apply_theme(app, saved_theme_name())

    win = MainWindow()

    def show_main():
        win.show()
        # Defer music start by one event-loop tick so any splash teardown
        # finishes first (two simultaneous QMediaPlayer streams during splash
        # close was crashing the app).
        from PySide6.QtCore import QTimer
        QTimer.singleShot(150, win.start_music)

    # Show splash if SplashScreen.mp4 is present, then open main window when video ends or user clicks/keypress.
    # If no splash file, show main window immediately.
    splash = maybe_show_splash(show_main)
    if splash is None:
        # Already shown by callback above
        pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
