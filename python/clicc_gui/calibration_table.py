from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QHBoxLayout, QComboBox, QMessageBox, QFileDialog
)
import numpy as np
import json


class CalibrationTableDialog(QDialog):
    def __init__(self, Q=None, tau=None, std=None, D_set=None, parent=None):
        super().__init__(parent)
        self.D_set = D_set

        self.fit_type = D_set.get("fit_type", "Oxygen")

        titles = {
            "Oxygen": "Oxygen Calibration Data",
            "Temperature": "Temperature Calibration Data",
            "pH": "pH Calibration Data"
        }
        self.setWindowTitle(titles.get(self.fit_type, "Calibration Data"))

        layout = QVBoxLayout(self)

        self.Q_base = np.asarray(Q, dtype=float)  
        self.tau = np.asarray(tau, dtype=float)
        self.std = np.asarray(std, dtype=float) if std is not None else np.array([])

        self.unit_combo = QComboBox()

        if self.fit_type == "Oxygen":
            self.unit_combo.addItems(["% air", "kPa", "mmol/L"])
            self.unit_combo.setCurrentText(
                self.D_set.get("oxygen_unit", "% air")
            )

        elif self.fit_type == "Temperature":
            self.unit_combo.addItems(["°C", "K"])
            self.unit_combo.setCurrentText(
                self.D_set.get("temperature_unit", "°C")
            )

        elif self.fit_type == "pH":
            self.unit_combo.hide()

        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)

        layout.addWidget(self.unit_combo)

        self.table = QTableWidget(0, 3)
        layout.addWidget(self.table)

        self.set_calibration_type(self.fit_type)

        if Q is not None and tau is not None:
            self.Q_base = np.asarray(Q, dtype=float)
            self.tau = np.asarray(tau, dtype=float)
            self.refresh_table()

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add row")
        rm_btn = QPushButton("Remove row")
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        btn_save = QPushButton("Save calibration…")
        btn_load = QPushButton("Load calibration…")

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(rm_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_load)
        btn_layout.addStretch()

        add_btn.clicked.connect(self.add_row)
        rm_btn.clicked.connect(self.remove_row)
        apply_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_save.clicked.connect(self.save_calibration)
        btn_load.clicked.connect(self.load_calibration)

    def add_row(self, Q_val="", tau_val="", std_val=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(Q_val)))
        self.table.setItem(row, 1, QTableWidgetItem(str(tau_val)))
        self.table.setItem(row, 2, QTableWidgetItem(str(std_val)))

    def remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def get_data(self):
        Q_disp = []
        tau = []
        std = []

        unit = self.unit_combo.currentText()

        for row in range(self.table.rowCount()):
            try:
                q = float(self.table.item(row, 0).text())
                t = float(self.table.item(row, 1).text())
                s = float(self.table.item(row, 2).text())
                Q_disp.append(q)
                tau.append(t)
                std.append(s)
            except:
                continue

        Q_disp = np.asarray(Q_disp, dtype=float)

        if self.fit_type == "Temperature":
            if unit == "°C":
                Q_base = Q_disp + 273.15
            else:
                Q_base = Q_disp
        else:
            Q_base = Q_disp

        return Q_base, np.asarray(tau), np.asarray(std)


    def save_calibration(self):
        from PyQt6.QtWidgets import QFileDialog
        import json
        import numpy as np
        

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save calibration data",
            "",
            "Calibration files (*.json)"
        )
        if not path:
            return

        Q, tau, std = self.get_data()

        data = {
            "fit_type": self.fit_type,
            "Q": np.asarray(Q, dtype=float).tolist(),
            "tau": np.asarray(tau, dtype=float).tolist(),
            "std": np.asarray(std, dtype=float).tolist()
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        QMessageBox.information(self, "Saved", f"Calibration saved to:\n{path}")
    
    def load_calibration(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load calibration data",
            "",
            "Calibration files (*.json)"
        )
        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        Q   = np.asarray(data.get("Q", []), dtype=float)
        tau = np.asarray(data.get("tau", []), dtype=float)
        std = np.asarray(data.get("std", []), dtype=float)

        if len(std) != len(Q):
            std = np.zeros_like(Q)

        self.set_data(Q, tau, std)

    def set_data(self, Q, tau, std=None):
        self.Q_base = np.asarray(Q, dtype=float)
        self.tau = np.asarray(tau, dtype=float)

        if std is None:
            self.std = np.zeros_like(self.tau)
        else:
            self.std = np.asarray(std, dtype=float)

        self.refresh_table()
        
    def set_calibration_type(self, calib_type: str):

        if calib_type == "Oxygen":
            unit = self.unit_combo.currentText()
            self.table.setHorizontalHeaderLabels([
                f"Oxygen ({unit})",
                "Lifetime τ (ms)",
                "STD τ (ms)"
            ])

        elif calib_type == "Temperature":
            unit = self.unit_combo.currentText()
            self.table.setHorizontalHeaderLabels([
                f"Temperature ({unit})",
                "Lifetime τ (ms)",
                "STD τ (ms)"
            ])

        elif calib_type == "pH":
            self.table.setHorizontalHeaderLabels([
                "pH",
                "Ratio (G2 / G1)",
                "STD ratio"
            ])

    def convert_from_base(self, Q_base, unit):

        Q_base = np.asarray(Q_base, dtype=float)

        if self.fit_type == "Oxygen":
            return Q_base   

        elif self.fit_type == "Temperature":
            if unit == "°C":
                return Q_base - 273.15
            return Q_base   # Kelvin

        elif self.fit_type == "pH":
            return Q_base
    
    def convert_to_base(self, Q_display, unit):

        Q_display = np.asarray(Q_display, dtype=float)

        if self.fit_type == "Oxygen":
            return Q_display

        elif self.fit_type == "Temperature":
            if unit == "°C":
                return Q_display + 273.15
            return Q_display

        elif self.fit_type == "pH":
            return Q_display

    def on_unit_changed(self, unit):

        if self.fit_type == "Oxygen":
            self.D_set["oxygen_unit"] = unit

        elif self.fit_type == "Temperature":
            self.D_set["temperature_unit"] = unit

        self.refresh_table()
    
    def refresh_table(self):
        if self.fit_type == "Temperature":
            unit = self.unit_combo.currentText()
            if unit == "°C":
                Q_disp = self.Q_base - 273.15
            else:
                Q_disp = self.Q_base.copy()
        else:
            Q_disp = self.Q_base.copy()

        if self.std.size != self.tau.size:
            self.std = np.zeros_like(self.tau)

        self.table.setRowCount(len(Q_disp))

        for i, (q, t, s) in enumerate(zip(Q_disp, self.tau, self.std)):
            self.table.setItem(i, 0, QTableWidgetItem(f"{q:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{t:.6g}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{s:.6g}"))

        self.update_header()
    
    def update_header(self):
        self.set_calibration_type(self.fit_type)