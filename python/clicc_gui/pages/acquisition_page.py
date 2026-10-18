import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ximea import xiapi

from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QGridLayout, QHBoxLayout,
    QGroupBox, QFileDialog, QSlider, QComboBox,
    QSizePolicy)

from PyQt6.QtGui import QPixmap, QIcon, QAction
from PyQt6.QtCore import Qt, QTimer

from matplotlib.widgets import RectangleSelector

from threads import AcquisitionThread, LEDTestThread
from globals_config import GUI_params
from globals_config import D_set

BASE_DIR = Path(__file__).resolve().parent.parent

class AcquisitionGUI(QWidget):
    def __init__(self, run_function, D_set):
        super().__init__()
        self.run_function = run_function
        self.D_set = D_set

        # MAIN LAYOUT
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        outer_group = QGroupBox("Acquisition Timing Scheme")
        outer_layout = QHBoxLayout()
        outer_group.setLayout(outer_layout)

        # LEFT SIDE
        left_group = QWidget()
        left_layout = QVBoxLayout()
        left_group.setLayout(left_layout)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pix = QPixmap(str(BASE_DIR / "timing_diagram.png"))
        self.image_label.setPixmap(pix.scaledToWidth(600, Qt.TransformationMode.SmoothTransformation))

        left_layout.addWidget(self.image_label)
        left_layout.addStretch(1)

        save_group = QGroupBox("Save Location")
        save_layout = QGridLayout()

        save_layout.addWidget(QLabel("Folder:"), 0, 0)
        self.dir_edit = QLineEdit(self.D_set['dir_path'])
        save_layout.addWidget(self.dir_edit, 0, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.choose_directory)
        save_layout.addWidget(browse_btn, 0, 2)

        save_layout.addWidget(QLabel("File name (.h5):"), 1, 0)
        self.file_edit = QLineEdit("measurement.h5")
        save_layout.addWidget(self.file_edit, 1, 1)

        self.naming_hint = QLabel(
            "&nbsp;&nbsp;For automatic calibration follow the filename convention:<br>"
            "&nbsp;&nbsp;<b>Temperature</b> → temp25 | temperature_25 | t25<br>"
            "&nbsp;&nbsp;<b>Oxygen</b> → ox10 | o2_10 | oxygen10<br>"
            "&nbsp;&nbsp;<b>pH</b> → ph7_4"
        )
        self.naming_hint.setStyleSheet("color: #8a8a8a; font-size: 10px;")
        self.naming_hint.setWordWrap(True)

        save_layout.addWidget(self.naming_hint, 2, 1)

        self.btn_save_params = QPushButton("Save parameters")
        self.btn_load_params = QPushButton("Load parameters")

        self.btn_save_params.clicked.connect(self.save_parameters)
        self.btn_load_params.clicked.connect(self.load_parameters)

        save_layout.addWidget(self.btn_save_params, 4, 0, 1, 3)
        save_layout.addWidget(self.btn_load_params, 5, 0, 1, 3)

        save_layout.setVerticalSpacing(6)

        self.dir_edit.editingFinished.connect(self.update_save_location)
        self.file_edit.editingFinished.connect(self.update_save_location)

        save_group.setLayout(save_layout)
        left_layout.addWidget(save_group)
        left_layout.addStretch(1)
        left_group.setLayout(left_layout)

        # RIGHT SIDE 
        right_group = QGroupBox("Acquisition Parameters")
        right_layout = QGridLayout()

        right_layout.setVerticalSpacing(17)
        right_layout.setHorizontalSpacing(12)

        row = 0

        self.material_box = QLineEdit()
        right_layout.addWidget(QLabel("Material:"), row, 0)
        right_layout.addWidget(self.material_box, row, 1)

        self.fps_label = QLabel("Expected FPS: --")
        right_layout.addWidget(self.fps_label, row, 2)

        self.temp_label = QLabel("Temperature (live): -- °C")
        right_layout.addWidget(self.temp_label, row, 3)
        row += 1

        self.mode_box = QComboBox()
        self.mode_box.addItems(["RLD", "RLD (overlapping)", "DLR"])

        right_layout.addWidget(QLabel("Mode:"), row, 0)
        right_layout.addWidget(self.mode_box, row, 1)

        self.fit_type_box = QComboBox()
        self.fit_type_box.addItems(["Oxygen", "Temperature", "pH"])
        self.fit_type_box.setCurrentText(self.D_set.get("fit_type", "Oxygen"))
        self.fit_type_box.setStyleSheet("QComboBox { color: #a184f5; }")

        right_layout.addWidget(QLabel("Measurement type:"), row, 2)
        right_layout.addWidget(self.fit_type_box, row, 3)
        row += 1

        self.mode_box.currentTextChanged.connect(self.update_mode)
        self.fit_type_box.currentTextChanged.connect(self.on_fit_type_changed)

        # Row 1
        self.delta_t_box = self.add_field_to_column(
            right_layout, "\u0394t (µs):",
            GUI_params["delta_t"] * 1e6,
            row, 0,
            hint="≈ 0.5–1 × maximum lifetime"
        )
        self.number_exposures_box = self.add_field_to_column(
            right_layout, "Number of Exposures:",
            GUI_params["Exposures"],
            row, 1
        )
        row += 1

        # Row 2
        self.t2_box = self.add_field_to_column(
            right_layout, "t2_dlr (µs):",
            GUI_params["t2_dlr"] * 1e6,
            row, 0
        )
        self.cam_delay_box = self.add_field_to_column(
            right_layout,
            "Time betw. Exposures (µs):",
            GUI_params["Time betw. Exposures"] * 1e6,
            row, 1,
            hint="≈ 5 × maximum lifetime"
        )
        row += 1

        # Row 3
        self.t_on_1_box = self.add_field_to_column(
            right_layout, "t_led_on (µs):",
            GUI_params["t_on_1"] * 1e6,
            row, 0
        )
        self.image_number_box = self.add_field_to_column(
            right_layout, "Number of Images:",
            GUI_params["Number of Images"],
            row, 1
        )
        row += 1

        # Row 4
        self.delay_led_box = self.add_field_to_column(
            right_layout, "t1_start (µs):",
            GUI_params["t1_start"] * 1e6,
            row, 0
        )
        self.cycle_delay_box = self.add_field_to_column(
            right_layout, "Cycle duration (s):",
            GUI_params["cycle_delay"],
            row, 1
        )
        row += 1

        # Row 5
        self.t2_start_box = self.add_field_to_column(
            right_layout, "t2_start (µs):",
            GUI_params["t2_start"] * 1e6,
            row, 0
        )

        self.slider_vval = self.add_slider_to_column(
            right_layout,
            label="LED-Voltage",
            min_val=24,
            max_val=49,
            value=int(24 + (self.D_set["vval"] / 4095) * (49 - 24)),
            row=row,
            column=1
        )
        row += 1

        self.t2_box.setDisabled(True)
        self.t2_start_box.setDisabled(True)

        self.delta_t_box.editingFinished.connect(self.update_parameters)
        self.t_on_1_box.editingFinished.connect(self.update_parameters)
        self.t2_box.editingFinished.connect(self.update_parameters)
        self.delay_led_box.editingFinished.connect(self.update_parameters)
        self.number_exposures_box.editingFinished.connect(self.update_parameters)
        self.image_number_box.editingFinished.connect(self.update_parameters)
        self.cam_delay_box.editingFinished.connect(self.update_parameters)
        self.t2_start_box.editingFinished.connect(self.update_parameters)
        self.cycle_delay_box.editingFinished.connect(self.update_parameters)
        self.material_box.editingFinished.connect(self.update_material)

        self.slider_vval.valueChanged.connect(self.update_vval)

        # BUTTONS
        start_btn = QPushButton("Start Acquisition")
        start_btn.setFixedHeight(40)
        start_btn.setStyleSheet("color: #a184f5;")
        start_btn.clicked.connect(self.start_acquisition)

        right_layout.addWidget(start_btn, row, 0, 1, 4)

        right_layout.setRowStretch(0, 1)        
        right_layout.setRowStretch(row + 1, 1)  

        for r in range(right_layout.rowCount()):
            right_layout.setRowMinimumHeight(r, 24)

        INPUT_WIDTH = 140
        right_layout.setColumnMinimumWidth(1, INPUT_WIDTH)
        right_layout.setColumnMinimumWidth(3, INPUT_WIDTH)

        wrapper_layout = QVBoxLayout()
        wrapper_layout.addStretch()         
        wrapper_layout.addLayout(right_layout)
        wrapper_layout.addStretch()       

        right_group.setLayout(wrapper_layout)

        outer_layout.addWidget(left_group, stretch=2)

        right_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        right_wrapper = QVBoxLayout()
        right_wrapper.addStretch()
        right_wrapper.addWidget(right_group)
        right_wrapper.addStretch()

        outer_layout.addLayout(right_wrapper, stretch=3)
        main_layout.addWidget(outer_group)

        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
    
    def _sanitize_for_json(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return float(obj)
        return obj
    
    def save_parameters(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save acquisition parameters",
            self.D_set["dir_path"],
            "JSON files (*.json)"
        )
        if not path:
            return

        data = {
            "folder": self.dir_edit.text(),
            "file_name": self.file_edit.text(),
            "material": self.material_box.text(),
            "mode": self.mode_box.currentText(),
            "fit_type": self.fit_type_box.currentText(),

            "delta_t": float(self.delta_t_box.text()),
            "t2_dlr": float(self.t2_box.text()),
            "t_led_on": float(self.t_on_1_box.text()),
            "t1_start": float(self.delay_led_box.text()),
            "t2_start": float(self.t2_start_box.text()),

            "time_between_exp": float(self.cam_delay_box.text()),
            "exposures": int(self.number_exposures_box.text()),
            "num_images": int(self.image_number_box.text()),
            "cycle_delay": float(self.cycle_delay_box.text()),

            "led_voltage": self.slider_vval.value(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        QMessageBox.information(self, "Saved", "Acquisition parameters saved.")

    def load_parameters(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load acquisition parameters",
            self.D_set["dir_path"],
            "JSON files (*.json)"
        )
        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.dir_edit.setText(
            data.get("folder", self.dir_edit.text())
        )
        self.file_edit.setText(
            data.get("file_name", self.file_edit.text())
        )

        self.material_box.setText(
            data.get("material", self.material_box.text())
        )

        if "mode" in data:
            self.mode_box.setCurrentText(data["mode"])

        if "fit_type" in data:
            self.fit_type_box.setCurrentText(data["fit_type"])

        if "delta_t" in data:
            self.delta_t_box.setText(str(data["delta_t"]))

        if "t2_dlr" in data:
            self.t2_box.setText(str(data["t2_dlr"]))

        if "t_led_on" in data:
            self.t_on_1_box.setText(str(data["t_led_on"]))

        if "t1_start" in data:
            self.delay_led_box.setText(str(data["t1_start"]))

        if "t2_start" in data:
            self.t2_start_box.setText(str(data["t2_start"]))

        if "time_between_exp" in data:
            self.cam_delay_box.setText(str(data["time_between_exp"]))

        if "exposures" in data:
            self.number_exposures_box.setText(str(data["exposures"]))

        if "num_images" in data:
            self.image_number_box.setText(str(data["num_images"]))

        if "cycle_delay" in data:
            self.cycle_delay_box.setText(str(data["cycle_delay"]))

        if "led_voltage" in data:
            self.slider_vval.setValue(int(data["led_voltage"]))

        self.update_save_location()
        self.update_material()
        self.update_parameters()
        self.update_vval(self.slider_vval.value())

        QMessageBox.information(self, "Loaded", "Acquisition parameters loaded.")
    
    def add_field_to_column(self, layout, label, value, row, column, hint=None):
        lab = QLabel(label)

        box = QLineEdit(str(value))
        box.setFixedHeight(22)

        if hint:
            action = QAction(QIcon(str(BASE_DIR / "exclamation_mark.png")), "", box)
            action.setToolTip(hint)

            box.addAction(action, QLineEdit.ActionPosition.TrailingPosition)

        layout.addWidget(lab, row, column*2)
        layout.addWidget(box, row, column*2 + 1)

        return box
    
    def add_slider_to_column(self, layout, label, min_val, max_val, value, row, column):
        lab = QLabel(f"{label}: {value}")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)

        slider.valueChanged.connect(lambda v: lab.setText(f"{label}: {v} V"))

        layout.addWidget(lab, row, column*2)
        layout.addWidget(slider, row, column*2 + 1)
        return slider
    
    def update_vval(self, slider_value):
        vval_bits = int((slider_value - 24) / (49 - 24) * 4095)
        self.D_set["vval"] = vval_bits

    def update_mode(self, mode):
        self.D_set["mode"] = mode

        if mode == "RLD":
            self.t2_start_box.setDisabled(True)  
            self.t2_box.setDisabled(True)  
            self.update_image(str(BASE_DIR / "timing_diagram.png"))
            self.D_set["delay_cam_end"] = round(GUI_params["Time betw. Exposures"] - 2 * self.D_set["delta_t"] - self.D_set["delay_led_end"], 9)
            self.D_set['t_G1'] = self.D_set['t_on_1'] + self.D_set['delay_led_end'] + self.D_set['delay_cam_end'] + 2*self.D_set['delta_t']
            self.D_set['t_G2'] = self.D_set['t_G1']

        elif mode == "RLD (overlapping)":
            self.t2_start_box.setDisabled(False)  
            self.t2_box.setDisabled(True)  
            self.update_image(str(BASE_DIR / "timing_diagram_overlapping.png"))
            self.D_set["delay_cam_end"] = round(GUI_params["Time betw. Exposures"] - 2 * self.D_set["delta_t"] - self.D_set["delay_led_end"], 9)
            self.D_set['t_G1'] = self.D_set['t_on_1'] + self.D_set['delay_led_end'] + self.D_set['delay_cam_end'] + 2*self.D_set['delta_t']
            self.D_set['t_G2'] = self.D_set['t_G1']

        elif mode == "DLR":
            self.t2_start_box.setDisabled(True)
            self.t2_box.setDisabled(False)  
            self.update_image(str(BASE_DIR / "timing_diagram_dlr.png"))
            self.D_set["delay_cam_end"] = round(GUI_params["Time betw. Exposures"] - self.D_set["t2_dlr"] - self.D_set["delay_led_end"], 9) 
            self.D_set['t_G1'] = self.D_set['t_on_1'] + self.D_set['delay_cam_end'] + self.D_set['t2_dlr'] + self.D_set['delay_led_end']
            self.D_set['t_G2'] = self.D_set['t_on_1'] + self.D_set['delay_led_end'] + self.D_set['t2_dlr'] + self.D_set['delay_cam_end']

        self.D_set['image_acq_time1'] = self.D_set['number_exposure'] * self.D_set['t_G1'] + self.D_set['readout']
        self.D_set['image_acq_time2'] = self.D_set['number_exposure'] * self.D_set['t_G2'] + self.D_set['readout']
        self.D_set['fps'] = 1/(self.D_set['number_exposure']*(self.D_set['t_G1']+self.D_set['t_G2'])+2*self.D_set['readout'])

        self.update_expected_fps()
    
    def on_fit_type_changed(self, fit_type):
        self.D_set["fit_type"] = fit_type

        if hasattr(self, "btn_measure_ref"):
            self.btn_measure_ref.setVisible(fit_type == "pH")

        main_window = self.window()

        if hasattr(main_window, "calibration_page"):
            main_window.calibration_page.update_range_mode()

        if hasattr(main_window, "analysis_page"):
            main_window.analysis_page.update_range_mode()
            main_window.analysis_page.set_default_analysis_range()
            main_window.analysis_page.update_analysis_mode_visibility()

        if hasattr(main_window, "fit_page"):
            main_window.fit_page.update_parameter_visibility()

    def update_image(self, img_file):
        self.pixmap = QPixmap(img_file)
        self.image_label.setPixmap(self.pixmap.scaledToWidth(600, Qt.TransformationMode.SmoothTransformation))

    def update_save_location(self):
        self.D_set['dir_path'] = self.dir_edit.text()
        self.D_set['name'] = self.file_edit.text()

        if not self.D_set['name'].lower().endswith(".h5"):
            self.D_set['name'] += ".h5"
            self.file_edit.setText(self.D_set['name'])

        print(f"Save location set to: {self.D_set['dir_path']}\\{self.D_set['name']}")
    
    def choose_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.dir_edit.text())
        if folder:
            self.dir_edit.setText(folder)
            self.D_set['dir_path'] = folder
            self.update_save_location()
        
    def update_temperature(self):
        if not hasattr(self, "cam"):
            self.temp_label.setText("Temperature (live): -- °C")
            return

        try:
            temp = self.cam.get_param(xiapi.XI_PRM_TEMP)
            self.temp_label.setText(f"Temperature (live): {temp:.1f} °C")
        except:
            self.temp_label.setText("Temperature (live): n/a")
    
    def update_expected_fps(self):
        try:
            cd = self.D_set.get("cycle_delay", 0)

            if cd > 0:
                fps = 1 / cd
                self.fps_label.setText(f"Expected FPS: {fps:.2f}")
                return

            n_exp = self.D_set["number_exposure"]
            tG1 = self.D_set["t_G1"]
            tG2 = self.D_set["t_G2"]
            readout = self.D_set["readout"]

            fps = 1 / (n_exp * (tG1 + tG2) + 2 * readout)
            self.fps_label.setText(f"Expected FPS: {fps:.2f}")

        except Exception as e:
            self.fps_label.setText("Expected FPS: --")
            print("FPS update error:", e)
        
    def update_material(self):
        material = self.material_box.text().strip()
        if material == "":
            material = "Unknown"

        self.D_set["name_phosphor"] = material

    def update_parameters(self):
        try:
            GUI_params["delta_t"] = round(float(self.delta_t_box.text()) * 1e-6, 9)
            GUI_params["t2_dlr"] = round(float(self.t2_box.text()) * 1e-6, 9)
            GUI_params["t_on_1"] = round(float(self.t_on_1_box.text()) * 1e-6, 9)
            GUI_params["t1_start"] = float(self.delay_led_box.text()) * 1e-6
            GUI_params["Exposures"] = int(self.number_exposures_box.text())
            GUI_params["Number of Images"] = int(self.image_number_box.text())
            GUI_params["Time betw. Exposures"] = float(self.cam_delay_box.text()) * 1e-6
            GUI_params["cycle_delay"] = float(self.cycle_delay_box.text())
            GUI_params["t2_start"] = round(float(self.t2_start_box.text()) * 1e-6, 9)

            self.D_set["delta_t"] = GUI_params["delta_t"]
            self.D_set["t2_dlr"] = GUI_params["t2_dlr"]
            self.D_set["t_on_1"] = GUI_params["t_on_1"]
            self.D_set["delay_led_end"] = GUI_params["t1_start"]
            self.D_set["number_exposure"] = GUI_params["Exposures"]
            self.D_set["nbr_shots_total"] = GUI_params["Number of Images"]
            self.D_set["cycle_delay"] = GUI_params["cycle_delay"]
            self.D_set["t2_start"] = GUI_params["t2_start"]

            if self.D_set["t2_start"] > self.D_set["delta_t"]:
                self.D_set["t2_start"] = self.D_set["delta_t"]
                self.t2_start_box.setText(f"{self.D_set['t2_start'] * 1e6:.0f}")

                QMessageBox.warning(
                    self,
                    "Value constraint",
                    "t2_start must be ≤ Δt.\n"
                    f"It has been automatically set to {self.D_set['t2_start']*1e6:.0f} µs."
                )

            T_exp = GUI_params["Time betw. Exposures"]
            margin = 20e-6

            t1_start = self.D_set["delay_led_end"]  
            delta_t = self.D_set["delta_t"]
            t2_start = self.D_set["t2_start"]
            t2_dlr = self.D_set["t2_dlr"]

            if self.D_set["mode"] == "RLD":
                T_min = t1_start + 2 * delta_t + margin

            elif self.D_set["mode"] == "RLD (overlapping)":
                if t2_start != 0:
                    T_min = t1_start + delta_t + t2_start + margin
                else:
                    T_min = t1_start + 2 * delta_t + margin

            elif self.D_set["mode"] == "DLR":
                T_min = t1_start + t2_dlr + margin

            else:
                T_min = 0
            
            if T_exp < T_min:
                QMessageBox.warning(
                    self,
                    "Timing constraint",
                    f"Time betw. Exposures is too short.\n\n"
                    f"It has been increased to {T_min*1e6:.0f} µs."
                )

                T_exp = T_min
                GUI_params["Time betw. Exposures"] = T_exp
                self.cam_delay_box.setText(f"{T_exp*1e6:.0f}")

            if self.D_set["mode"] in ["RLD", "RLD (overlapping)"]:
                self.D_set["delay_cam_end"] = round(
                    T_exp - 2 * delta_t - t1_start,
                    9
                )

            elif self.D_set["mode"] == "DLR":
                self.D_set["delay_cam_end"] = round(
                    T_exp - t2_dlr - t1_start,
                    9
                )

            if self.D_set["mode"] in ["RLD", "RLD (overlapping)"]:
                self.D_set["t_G1"] = self.D_set["t_on_1"] + self.D_set["delay_led_end"] + self.D_set["delay_cam_end"] + 2 * delta_t
                self.D_set["t_G2"] = self.D_set["t_G1"]

            elif self.D_set["mode"] == "DLR":
                self.D_set["t_G1"] = self.D_set["t_on_1"] + self.D_set["delay_led_end"] + self.D_set["delay_cam_end"] + t2_dlr
                self.D_set["t_G2"] = self.D_set["t_G1"]

            self.D_set["t2_start"] = GUI_params["t2_start"]
            self.D_set['image_acq_time1'] = self.D_set['number_exposure'] * self.D_set['t_G1'] + self.D_set['readout']
            self.D_set['image_acq_time2'] = self.D_set['number_exposure'] * self.D_set['t_G2'] + self.D_set['readout']
            self.D_set['fps'] = 1/(self.D_set['number_exposure']*(self.D_set['t_G1']+self.D_set['t_G2'])+2*self.D_set['readout'])
            self.update_expected_fps()

        except ValueError:
            QMessageBox.critical(self, "Error", "Please enter numeric values.")
    
    def show_led_test_with_roi_selector(self, img):
        self._led_test_img = img  

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(img, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title("Select ROI")

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            x0, x1 = sorted([x1, x2])
            y0, y1 = sorted([y1, y2])

            roi = self._led_test_img[y0:y1, x0:x1]

            if roi.size == 0:
                print("Empty ROI selected")
                return

            mean_val = float(np.nanmean(roi))
            std_val = float(np.nanstd(roi))
            cv = std_val / mean_val if mean_val > 0 else np.nan

            sat_pct = float(np.mean(roi >= 4094) * 100)  

            msg = (
                f"Mean: {mean_val:.2f}\n"
                f"Std: {std_val:.2f}\n"
                f"Coefficient of variation: {cv:.4f}\n"
                f"Saturation %: {sat_pct:.2f}"
            )

            QMessageBox.information(self, "LED ROI Metrics", msg)

            self.D_set["led_roi"] = (x0, y0, x1 - x0, y1 - y0)

        self._profile_selector = RectangleSelector(
        ax,
        onselect=onselect,
        useblit=True,
        interactive=False,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        props=dict(edgecolor='cornflowerblue', facecolor='none', linewidth=1.5)
    )

        plt.show()

        
    def run_led_test(self):
        self.btn_led_test.setEnabled(False)

        self.led_thread = LEDTestThread(self.D_set)
        self.led_thread.finished.connect(self.on_led_test_done)
        self.led_thread.error.connect(self.on_led_test_error)
        self.led_thread.start()
    
    def on_led_test_done(self, img, metrics):
        self.btn_led_test.setEnabled(True)
        self.show_led_test_with_roi_selector(img)

    def on_led_test_error(self, msg):
        self.btn_led_test.setEnabled(True)
        QMessageBox.critical(self, "LED Test Error", msg)

    def start_acquisition(self):

        dir_path = self.D_set.get("dir_path", "")
        filename = self.D_set.get("name", "")
        full_path = os.path.join(dir_path, filename)

        if os.path.exists(full_path):
            reply = QMessageBox.question(
                self,
                "File Error",
                f"The file '{full_path}' already exists.\n\n"
                "Do you want to overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                QMessageBox.information(self, "Cancelled", "Acquisition aborted. Choose a new filename.")
                return

        self.final_temperature = None 
        self.temp_label.setText("Temperature (live): -- °C")

        self.temp_timer.start(1000)

        self.D_set["parameters_to_send"] = (
            f"{self.D_set['number_exposure']},"
            f"{int(self.D_set['delta_t']*1e6)},"
            f"{int(self.D_set['t2_dlr']*1e6)},"
            f"{int(self.D_set['t_on_1']*1e6)},"
            f"{int(self.D_set['delay_led_end']*1e6)},"
            f"{int(self.D_set['delay_cam_end']*1e6)},"
            f"{self.D_set['vval']},"
            f"{int(self.D_set['t2_start']*1e6)}"
        )

        print(f"Sending parameters: {self.D_set['parameters_to_send']}")

        self.thread = AcquisitionThread(lambda: self.run_function(self))
        self.thread.error_signal.connect(self.on_error)
        self.thread.done_signal.connect(self.on_done)
        self.thread.start()

    def on_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def on_done(self):

        self.temp_timer.stop()

        if hasattr(self, "final_temperature") and self.final_temperature is not None:
            self.temp_label.setText(
                f"Temperature (end of acq.): {self.final_temperature:.1f} °C"
            )
        else:
            
            self.temp_label.setText("Temperature (end of acq.): -- °C")

        QMessageBox.information(self, "Done", "Acquisition completed successfully!")
    
    def measure_reference(self):
        reply = QMessageBox.question(
            self,
            "pH reference measurement",
            "Start reference measurement?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        base = self.file_edit.text().replace(".h5", "")
        ref_name = base + "_ref.h5"

        self.file_edit.setText(ref_name)

        self.start_acquisition()



