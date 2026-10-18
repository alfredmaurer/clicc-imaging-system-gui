import os

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QMessageBox, QHBoxLayout,
    QFileDialog, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

from threads import LiveCameraThread
from ximea import xiapi

class LiveViewGUI(QWidget):
    def __init__(self, D_set):
        super().__init__()
        self.D_set = D_set
        self.cam = None
        self.thread = None
        self.last_frame = None
        self.captured_frame = None

        layout = QVBoxLayout(self)

        self.image_label = QLabel("Live View ...")
        self.image_label.setMinimumSize(640, 520)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #121212;
                border: 1px solid #2b2b2b;
                border-radius: 6px;
                color: rgba(255, 255, 255, 60);   /* bleached via opacity */
                font-size: 13px;
            }
            """)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self.purple_button_style = """
            QPushButton {
                background-color: #1e1e1e;      /* dark grey */
                color: #a184f5;                 /* purple text */
                font-weight: bold;
                border: 1px solid #2b2b2b;
                border-radius: 6px;
                padding: 7px 16px;
            }

            QPushButton:hover {
                background-color: #262626;      /* slightly lighter */
            }

            QPushButton:pressed {
                background-color: #151515;      /* slightly darker */
            }

            QPushButton:disabled {
                background-color: #1e1e1e;
                color: #6f6790;                 /* muted purple */
                border-color: #252525;
            }
            """

        layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_save = QPushButton("Save Image")

        self.btn_start.setStyleSheet(self.purple_button_style)
        self.btn_stop.setStyleSheet(self.purple_button_style)
        self.btn_save.setStyleSheet(self.purple_button_style)

        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_save.clicked.connect(self.save_image)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

    def start_camera(self):
        try:
            self.cam = xiapi.Camera()
            self.cam.open_device()

            self.cam.set_exposure(10000)    
            self.cam.set_gain(0)
            self.cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FREE_RUN")

            self.cam.start_acquisition()

            self.thread = LiveCameraThread(self.cam)
            self.thread.new_frame.connect(self.update_frame)
            self.thread.error.connect(self.on_error)
            self.thread.start()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_save.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Camera error", str(e))

    def stop_camera(self):
        try:
            if self.thread:
                self.thread.stop()
                self.thread = None

            if self.cam:
                try:
                    self.cam.stop_acquisition()
                except:
                    pass

                self.cam.close_device()
                self.cam = None

        finally:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_save.setEnabled(False)

    def update_frame(self, frame):
        self.last_frame = frame.copy()  

        h, w = frame.shape

        qimg = QImage(
            frame.data,
            w,
            h,
            w,
            QImage.Format.Format_Grayscale8
        )

        pix = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        )
    
    def save_image(self):
        if self.last_frame is None:
            QMessageBox.warning(self, "No image", "No frame available.")
            return

        self.captured_frame = self.last_frame.copy()

        self.btn_save.setEnabled(False)

        try:

            default_dir = self.D_set.get("dir_path", os.getcwd())

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save snapshot",
                os.path.join(default_dir),
                "TIFF (*.tif *.tiff);;PNG (*.png);;All Files (*)"
            )

            if not path:
                return 

            img = self.captured_frame

            if img.dtype in (np.uint8, np.uint16):
                img_out = img
            else:
                img_out = img.astype(np.uint16)

            import imageio.v2 as imageio
            imageio.imwrite(path, img_out)

        finally:
            self.btn_save.setEnabled(True)

        QMessageBox.information(self, "Saved", f"Snapshot saved:\n{path}")


    def on_error(self, msg):
        QMessageBox.critical(self, "Live view error", msg)
        self.stop_camera()


