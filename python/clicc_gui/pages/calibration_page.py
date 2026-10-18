import os
import h5py

import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QLabel, QPushButton,
    QVBoxLayout, QMessageBox,
    QGroupBox, QFileDialog
)

from analysis_base import BaseImageAnalysisPage
import functions
from functions import compute_mean_ratio
from calibration_table import CalibrationTableDialog

class CalibrationGUI(BaseImageAnalysisPage):
    def __init__(self, D_set):
        super().__init__(D_set, title="Calibration Controls")

        self.calibration_Q = np.array([])
        self.calibration_tau = np.array([])

        self.btn_auto_calib = QPushButton("Automatic pH calibration")
        self.btn_auto_calib.clicked.connect(self.run_auto_pH_calibration)
        self.btn_auto_calib.setVisible(False)

        self.btn_auto_o2_calib = QPushButton("Automatic Oxygen calibration")
        self.btn_auto_o2_calib.clicked.connect(self.run_auto_O2_calibration)
        self.btn_auto_o2_calib.setVisible(False)

        self.btn_auto_temp_calib = QPushButton("Automatic Temperature calibration")
        self.btn_auto_temp_calib.clicked.connect(self.run_auto_temp_calibration)
        self.btn_auto_temp_calib.setVisible(False)

        self.btn_stats = QPushButton("Compute Lifetime in ROI")
        self.btn_stats.clicked.connect(self.compute_roi_stats)

        self.btn_save_fig = QPushButton("Save Figure")
        self.btn_save_fig.clicked.connect(self.save_figure)

        btn_save_roi = QPushButton("Save ROI")
        btn_save_roi.clicked.connect(self.save_roi)

        calib_group = QGroupBox("Calibration Table")
        calib_layout = QVBoxLayout(calib_group)

        self.btn_edit_calib = QPushButton("Edit calibration data")
        self.btn_edit_calib.clicked.connect(self.open_calibration_table)

        self.btn_show_calib_plot = QPushButton("Show calibration plot")
        self.btn_show_calib_plot.clicked.connect(self.show_calibration_plot)

        self.label_calib_points = QLabel("0 calibration points")

        calib_layout.addWidget(self.btn_edit_calib)
        calib_layout.addWidget(self.btn_show_calib_plot)
        calib_layout.addWidget(self.label_calib_points)

        self.page_controls.addWidget(self.btn_auto_calib)
        self.page_controls.addWidget(self.btn_auto_o2_calib)
        self.page_controls.addWidget(self.btn_auto_temp_calib)
        self.page_controls.addWidget(self.btn_stats)
        self.page_controls.addWidget(self.btn_save_fig)
        self.page_controls.addWidget(btn_save_roi)
        self.page_controls.addWidget(calib_group)

        self.refresh_button_text()

    def open_calibration_table(self):
        dlg = CalibrationTableDialog(
            Q=self.calibration_Q,
            tau=self.calibration_tau,
            std=getattr(self, "calibration_ratio_std", None),
            D_set=self.D_set,
            parent=self
        )

        if dlg.exec():
            self.calibration_Q, self.calibration_tau, self.calibration_ratio_std = dlg.get_data()

            self.label_calib_points.setText(
                f"{len(self.calibration_Q)} calibration points"
            )
    
    def set_axis_labels(self, ax, fit_type, unit="%"):
        if fit_type == "Oxygen":

            if unit == "%":
                ax.set_xlabel("Oxygen (%)")
            elif unit == "kPa":
                ax.set_xlabel("Oxygen (kPa)")
            elif unit == "mmol/L":
                ax.set_xlabel("Oxygen (mmol/L)")
            else:
                ax.set_xlabel("Oxygen")

            ax.set_ylabel("Lifetime τ (µs)")

        elif fit_type == "Temperature":
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Lifetime τ (µs)")

        elif fit_type == "pH":
            ax.set_xlabel("pH")
            ax.set_ylabel("Lifetime τ (µs)")


    def extract_pH_from_filename(self, path):
        import re
        name = os.path.basename(path).lower()

        m = re.search(
            r"ph\s*[-_]*\s*([0-9]+(?:[.,_][0-9]+)?)",
            name
        )

        if not m:
            return None

        try:
            val = m.group(1).replace(",", ".").replace("_", ".")
            pH = float(val)
        except ValueError:
            return None

        return pH

    def extract_oxygen_from_filename(self, path):
        import re
        name = os.path.basename(path).lower()

        # works for: ox5.h5, ox_5.h5, ox-5.5.h5, o2_21.h5, oxygen21.h5
        m = re.search(r"(?:ox|o2|oxygen)\s*[-_]*\s*([0-9]+(?:[.,][0-9]+)?)", name)

        if not m:
            return None

        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            return None


    def extract_temperature_from_filename(self, path):
        import re
        name = os.path.basename(path).lower()

        # works for: temp_25, t-25, temperature25
        m = re.search(r"(temp|temperature|t)\s*[-_]*\s*([0-9]+(?:[.,_][0-9]+)?)", name)

        if not m:
            return None

        try:
            val = m.group(2).replace(",", ".").replace("_", ".")
            return float(val)
        except ValueError:
            return None
    
    def compute_tau_avg_from_file(self, filename):
        params = functions.hdf5_read(filename, "parameters")
        delta_t = float(np.asarray(params["delta_t"]).item())

        with h5py.File(filename, "r") as f:
            groups_ratio = sorted([g for g in f.keys() if g.isdigit()])

        accum = None
        count = 0

        for g in groups_ratio:
            data = functions.hdf5_read(filename, g)
            Ratio = data["Ratio"].astype(np.float64)

            Ratio[(Ratio <= 0) | (Ratio >= 1)] = np.nan
            Ratio = np.clip(Ratio, 1e-6, 0.999999)

            tau = -delta_t / np.log(Ratio)
            tau_ms = 1000 * tau

            if accum is None:
                accum = np.zeros_like(tau_ms)

            accum += tau_ms
            count += 1

        if count == 0:
            raise ValueError("No valid Ratio frames found.")

        tau_avg = accum / count
        return tau_avg

    def compute_gate_means_from_file(self, filename):
        with h5py.File(filename, "r") as f:
            groups_gates = sorted([g for g in f.keys() if g.startswith("gates_")])

        G1_accum = None
        G2_accum = None
        gate_count = 0

        for g in groups_gates:
            data = functions.hdf5_read(filename, g)
            G1 = data["G1"].astype(np.float64)
            G2 = data["G2"].astype(np.float64)

            if G1_accum is None:
                G1_accum = np.zeros_like(G1)
                G2_accum = np.zeros_like(G2)

            G1_accum += G1
            G2_accum += G2
            gate_count += 1

        if gate_count == 0:
            raise ValueError("No gate frames found.")

        G1_avg = G1_accum / gate_count
        G2_avg = G2_accum / gate_count

        return G1_avg, G2_avg

        
    def run_auto_calibration_generic(self, mode="pH"):

        if self.roi is None:
            QMessageBox.warning(self, "ROI missing", "Please select an ROI first.")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            f"Select {mode} calibration measurements",
            self.D_set["dir_path"],
            "HDF5 files (*.h5)"
        )

        if not files:
            return

        added = 0
        skipped = []

        for path in files:

            if mode == "pH":
                Q_val = self.extract_pH_from_filename(path)

            elif mode == "Oxygen":
                Q_val = self.extract_oxygen_from_filename(path)

            elif mode == "Temperature":
                Q_val = self.extract_temperature_from_filename(path)
                if Q_val is not None:
                    Q_val += 273.15

            else:
                Q_val = None

            if Q_val is None:
                skipped.append(os.path.basename(path))
                continue

            try:
                xmin, xmax, ymin, ymax = self.roi

                # pH (ratio based)
                if mode == "pH":
                    ratio_map, _, _ = compute_mean_ratio(path)

                    roi_map = ratio_map[ymin:ymax, xmin:xmax]
                    mean_val = np.nanmean(roi_map)
                    std_val  = np.nanstd(roi_map)

                # Oxygen / Temperature 
                else:
                    tau_map = self.compute_tau_avg_from_file(path)

                    try:
                        thresh = float(self.thresh_box.text())
                    except Exception:
                        thresh = 80

                    G1_avg, G2_avg = self.compute_gate_means_from_file(path)

                    mask = (G1_avg < thresh) | (G2_avg < thresh)
                    tau_map = tau_map.copy()
                    tau_map[mask] = np.nan

                    roi_map = tau_map[ymin:ymax, xmin:xmax]
                    mean_val = np.nanmean(roi_map)
                    std_val  = np.nanstd(roi_map)

                if not np.isfinite(mean_val):
                    skipped.append(os.path.basename(path))
                    continue

                self.calibration_Q = np.append(self.calibration_Q, Q_val)
                self.calibration_tau = np.append(self.calibration_tau, mean_val)

                if not hasattr(self, "calibration_ratio_std"):
                    self.calibration_ratio_std = np.array([])

                self.calibration_ratio_std = np.append(self.calibration_ratio_std, std_val)

                added += 1

            except Exception as e:
                skipped.append(f"{os.path.basename(path)}  --> {str(e)}")
                continue

        self.label_calib_points.setText(f"{len(self.calibration_Q)} calibration points")

        msg = f"Added {added} calibration points."
        if skipped:
            msg += "\n\nSkipped files:\n" + "\n".join(skipped)

        QMessageBox.information(self, f"Automatic {mode} calibration", msg)

    def run_auto_pH_calibration(self):
        self.run_auto_calibration_generic(mode="pH")

    def run_auto_O2_calibration(self):
        self.run_auto_calibration_generic(mode="Oxygen")

    def run_auto_temp_calibration(self):
        self.run_auto_calibration_generic(mode="Temperature")
    
    def load_tau_map(self, path):
        with h5py.File(path, "r") as f:
            if "tau_avg" in f:
                return f["tau_avg"][:]
            elif "tau_map" in f:
                return f["tau_map"][:]
            else:
                raise ValueError("No lifetime map found in file (tau_avg or tau_map).")

        
    def refresh_button_text(self):
        fit_type = self.D_set.get("fit_type")

        self.btn_auto_calib.setVisible(fit_type == "pH")

        self.btn_auto_o2_calib.setVisible(fit_type == "Oxygen")
        self.btn_auto_temp_calib.setVisible(fit_type == "Temperature")

        if fit_type == "pH":
            self.btn_stats.setText("Compute Ratio in ROI")
        else:
            self.btn_stats.setText("Compute Lifetime in ROI")


    def show_calibration_plot(self):
        if self.calibration_Q.size == 0:
            QMessageBox.warning(
                self,
                "No calibration data",
                "No calibration data available.\n"
                "Please add calibration points first."
            )
            return

        fig, ax = plt.subplots(figsize=(6, 4))

        fit_type = self.D_set.get("fit_type", "Oxygen")

        if fit_type == "Temperature":
            unit = self.D_set.get("temperature_unit", "°C")

            if unit == "°C":
                Q_plot = self.calibration_Q - 273.15   # Kelvin → °C
            else:
                Q_plot = self.calibration_Q.copy()

        else:
            Q_plot = self.calibration_Q.copy()

        y = self.calibration_tau.copy()

        if fit_type != "pH":
            y = y * 1000

        yerr = None
        if hasattr(self, "calibration_ratio_std"):
            if len(self.calibration_ratio_std) == len(y):
                yerr = self.calibration_ratio_std.copy()

                if fit_type != "pH":
                    yerr = yerr * 1000  

        ax.errorbar(
            Q_plot,
            y,
            yerr=yerr,
            fmt="o",
            color="tab:blue",
            ecolor="black",
            capsize=4,
            markersize=6,
            label="Calibration data"
        )

        if fit_type == "Oxygen":
            unit = self.D_set.get("oxygen_unit", "%")
            self.set_axis_labels(ax, fit_type, unit)

        elif fit_type == "Temperature":
            unit = self.D_set.get("temperature_unit", "°C")
            ax.set_xlabel(f"Temperature ({unit})")
            ax.set_ylabel("Lifetime τ (ms)")

        elif fit_type == "pH":
            ax.set_xlabel("pH")
            ax.set_ylabel("Ratio (G2 / G1)")

        ax.set_title("Calibration Data")

        ax.title.set_size(16)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=12)

        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    def refresh_path(self):
        filename = self.D_set.get("name", "")
        if not filename.lower().endswith(".h5"):
            filename += ".h5"

        full_path = os.path.join(self.D_set["dir_path"], filename)
        self.current_file = full_path 
        self.file_label.setText(full_path)


