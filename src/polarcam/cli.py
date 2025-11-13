# src/polarcam/cli.py
from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication

def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)

    from polarcam.controller.controller import Controller
    from polarcam.app.main_window import MainWindow

    ctrl = Controller()
    w = MainWindow(ctrl)
    w.show()

    # Let this be the single shutdown path
    app.aboutToQuit.connect(w.safe_shutdown)

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
