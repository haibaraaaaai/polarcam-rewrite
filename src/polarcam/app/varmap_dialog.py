from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QSpinBox,
    QComboBox, QCheckBox, QPushButton, QProgressBar, QPlainTextEdit, QWidget
)


class VarMapDialog(QDialog):
    def __init__(self, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent,
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Variance map")
        self.resize(900, 650)

        self.ctrl = controller   # your Controller
        self._stack_path: Optional[Path] = None
        self._map_path: Optional[Path] = None
        self._last_map8: Optional[np.ndarray] = None
        self._cancelled: bool = False

        # --- Controls row ---
        self.spn_frames = QSpinBox()
        self.spn_frames.setRange(1, 1000)
        self.spn_frames.setValue(20)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "intensity_range",      # max - min (cheap)
            "stddev",               # per-pixel stdev (more robust)
            "max_pairwise_dist"     # largest |x_i - x_j| (placeholder/expensive)
        ])

        self.chk_memmap = QCheckBox("Memmap capture (low RAM)")
        self.chk_memmap.setChecked(True)

        self.btn_run = QPushButton("Capture + Compute")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        top = QHBoxLayout()
        top.addWidget(QLabel("Frames"))
        top.addWidget(self.spn_frames)
        top.addSpacing(12)
        top.addWidget(QLabel("Mode"))
        top.addWidget(self.cmb_mode)
        top.addSpacing(12)
        top.addWidget(self.chk_memmap)
        top.addStretch(1)
        top.addWidget(self.btn_run)
        top.addWidget(self.btn_cancel)

        # --- Progress ---
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        self.prog.setTextVisible(True)

        # --- Preview (variance map image) ---
        self.preview = QLabel("No result yet")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#111; color:#777;")
        self.preview.setMinimumSize(QSize(640, 360))

        # --- Log ---
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setStyleSheet("background:#0c0c0c; color:#d0d0d0; font-family: Consolas, monospace; font-size: 11px;")

        # --- Save row ---
        self.btn_save_map = QPushButton("Save map as .npy")
        self.btn_save_img = QPushButton("Save map as .png")
        self.btn_save_map.setEnabled(False)
        self.btn_save_img.setEnabled(False)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_row.addWidget(self.btn_save_map)
        save_row.addWidget(self.btn_save_img)

        # --- Layout ---
        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.prog)
        root.addWidget(self.preview, 1)
        root.addLayout(save_row)
        root.addWidget(self.log, 1)
        self.setLayout(root)

        # --- Hooks ---
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_save_map.clicked.connect(self._save_map_npy)
        self.btn_save_img.clicked.connect(self._save_map_png)

        # Place new window near parent, if any
        if parent is not None:
            g = parent.frameGeometry()
            self.move(g.center().x() - self.width() // 2, g.top() + 40)

    # ------------------------------------------
    # Public helpers
    # ------------------------------------------
    def append_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {text}")

    def set_progress(self, pct: int) -> None:
        self.prog.setValue(max(0, min(int(pct), 100)))

    def show_map(self, arr8u: np.ndarray) -> None:
        """Display an 8-bit map image (H,W) in the preview."""
        if arr8u is None or arr8u.ndim != 2 or arr8u.dtype != np.uint8:
            self.append_log("show_map: expected (H,W) uint8.")
            return
        self._last_map8 = arr8u
        h, w = arr8u.shape
        qimg = QImage(arr8u.data, w, h, w, QImage.Format_Grayscale8)
        self.preview.setPixmap(QPixmap.fromImage(qimg.copy()).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        # Keep preview scaled on resize
        pm = self.preview.pixmap()
        if pm:
            self.preview.setPixmap(pm.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(ev)

    def set_output_paths(self, stack_path: Optional[Path], map_path: Optional[Path]) -> None:
        self._stack_path = stack_path
        self._map_path = map_path
        have_map = map_path is not None
        self.btn_save_map.setEnabled(have_map)
        self.btn_save_img.setEnabled(have_map)

    # ------------------------------------------
    # Button handlers
    # ------------------------------------------
    def _on_run_clicked(self) -> None:
        n = int(self.spn_frames.value())
        mode = self.cmb_mode.currentText()
        use_memmap = self.chk_memmap.isChecked()

        self.log.clear()
        self.append_log(f"Capture {n} frame(s); mode={mode}; memmap={use_memmap}")
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_save_map.setEnabled(False)
        self.btn_save_img.setEnabled(False)
        self.set_progress(0)
        self._cancelled = False

        try:
            # Run synchronously; controller pumps events internally.
            res = self.ctrl.varmap_capture_and_compute(
                n_frames=n,
                mode=mode,
                memmap=use_memmap,            # controller accepts memmap/use_memmap
                on_progress=self._progress_cb,
                cancel_flag=lambda: self._cancelled,
            )

            # Show + enable saves
            self.append_log("Compute finished.")
            self.show_map(res["map8"])
            stack_path = Path(res["stack_path"]) if res.get("stack_path") else None
            map_path = Path(res["map_path"]) if res.get("map_path") else None
            self.set_output_paths(stack_path, map_path)
            if stack_path:
                self.append_log(f"Saved stack: {stack_path}")
            if map_path:
                self.append_log(f"Saved map:   {map_path}")

        except Exception as e:
            self.append_log(f"Error: {e}")
        finally:
            self.btn_run.setEnabled(True)
            self.btn_cancel.setEnabled(False)

    def _on_cancel_clicked(self) -> None:
        # We rely on cancel_flag() being polled in the controller loop
        self._cancelled = True
        self.append_log("Cancel requested.")
        self.btn_cancel.setEnabled(False)

    # ------------------------------------------
    # Progress callback passed to controller
    # ------------------------------------------
    def _progress_cb(self, frac: float, msg: str = "") -> None:
        self.set_progress(int(max(0.0, min(1.0, float(frac))) * 100))
        if msg:
            # Keep logs lightweight—only append occasionally if desired
            self.append_log(msg)

    # ------------------------------------------
    # Save helpers
    # ------------------------------------------
    def _save_map_npy(self) -> None:
        # Controller already saved a .npy; just log it.
        if self._map_path and self._map_path.suffix.lower() == ".npy":
            self.append_log(f"Map saved: {self._map_path}")
        else:
            self.append_log("Nothing to save yet.")

    def _save_map_png(self) -> None:
        if not (self._map_path and self._last_map8 is not None):
            self.append_log("Nothing to save yet.")
            return
        # Save a PNG alongside the .npy
        png_path = self._map_path.with_suffix(".png")
        arr = self._last_map8
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        if qimg.save(str(png_path)):
            self.append_log(f"PNG saved: {png_path}")
            # remember this too
            self._map_path = png_path
        else:
            self.append_log("Failed to save PNG.")
