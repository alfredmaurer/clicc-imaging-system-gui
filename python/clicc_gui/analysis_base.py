import os
import tempfile
from io import BytesIO

import numpy as np
import h5py
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QHBoxLayout,
    QGroupBox, QFileDialog, QComboBox,
    QSizePolicy, QInputDialog,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.widgets import RectangleSelector

import functions
from functions import compute_mean_ratio

TITLE_SIZE = 16
CBAR_SIZE = 12

class BaseImageAnalysisPage(QWidget):
    def __init__(self, D_set, title="Analysis Controls"):
        super().__init__()

        self.D_set = D_set
        self.roi = None
        self.allow_roi_popup = True

        filename = D_set.get("name", "")
        if not filename.lower().endswith(".h5"):
            filename += ".h5"

        self.current_file = os.path.join(D_set["dir_path"], filename)

        main_layout = QHBoxLayout(self)

        # LEFT SIDE 
        self.left_group = QGroupBox(title)
        self.left_layout = QVBoxLayout(self.left_group)

        self._build_file_selector()
        self._build_display_controls()
        self._build_roi_controls()

        self.page_controls = QVBoxLayout()
        self.left_layout.addLayout(self.page_controls)

        self.left_layout.addStretch()

        self._build_range_controls()

        main_layout.addWidget(self.left_group, stretch=1)

        # RIGHT SIDE 
        self.right_group = QGroupBox("Figure Display")
        right_layout = QVBoxLayout(self.right_group)

        self.fig_label = QLabel("Figures will appear here")
        self.fig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fig_label.setStyleSheet(
            "background: #EEEEEE; border: 1px solid #CCCCCC;"
        )
        self.fig_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored
        )

        right_layout.addWidget(self.fig_label)
        main_layout.addWidget(self.right_group, stretch=2)

        self.update_display_modes()
        self.update_range_mode()
        

    def _build_file_selector(self):
        row = QHBoxLayout()

        btn_load = QPushButton("Select File…")
        btn_load.clicked.connect(self.select_file)
        row.addWidget(btn_load)

        self.file_label = QLineEdit(self.current_file)
        row.addWidget(self.file_label)

        self.left_layout.addLayout(row)

    def _build_display_controls(self):
        row = QHBoxLayout()

        self.select_display = QComboBox()

        fit_type = self.D_set.get("fit_type", "Oxygen")

        if fit_type == "pH":
            self.select_display.addItems(
                ["All Images", "Ratio", "Gate 1", "Gate 2", "Background"]
            )
        else:
            self.select_display.addItems(
                ["All Images", "Lifetime", "Gate 1", "Gate 2", "Background"]
            )

        self.select_display.currentTextChanged.connect(self.on_mode_changed)
        row.addWidget(self.select_display)

        btn_fig = QPushButton("Show Figure")
        btn_fig.clicked.connect(self.on_show_figure_clicked)
        row.addWidget(btn_fig)

        btn_popout = QPushButton("Open full-resolution figure")
        btn_popout.clicked.connect(self.show_full_resolution_figure)
        row.addWidget(btn_popout)

        self.left_layout.addLayout(row)

    def _build_roi_controls(self):
        roi_group = QGroupBox("ROI Coordinates")
        layout = QHBoxLayout(roi_group)

        self.roi_xmin = QLineEdit("0")
        self.roi_xmax = QLineEdit("2464")
        self.roi_ymin = QLineEdit("0")
        self.roi_ymax = QLineEdit("2056")

        for box in (self.roi_xmin, self.roi_xmax, self.roi_ymin, self.roi_ymax):
            box.setFixedWidth(50)
            box.editingFinished.connect(self.update_roi_from_boxes)

        layout.addWidget(QLabel("xmin"))
        layout.addWidget(self.roi_xmin)
        layout.addWidget(QLabel("xmax"))
        layout.addWidget(self.roi_xmax)
        layout.addWidget(QLabel("ymin"))
        layout.addWidget(self.roi_ymin)
        layout.addWidget(QLabel("ymax"))
        layout.addWidget(self.roi_ymax)

        self.left_layout.addWidget(roi_group)

    def set_roi_from_dset(self):
        if "ROI" not in self.D_set:
            return

        xmin, xmax, ymin, ymax = self.D_set["ROI"]
        self.roi = (xmin, xmax, ymin, ymax)

        self.roi_xmin.setText(str(xmin))
        self.roi_xmax.setText(str(xmax))
        self.roi_ymin.setText(str(ymin))
        self.roi_ymax.setText(str(ymax))
        
    
    def _build_range_controls(self):

        tg = QGroupBox("Background Threshold")
        tl = QHBoxLayout(tg)

        self.thresh_box = QLineEdit("0")
        self.thresh_box.setFixedWidth(60)

        tl.addWidget(QLabel("Min intensity:"))
        tl.addWidget(self.thresh_box)

        self.left_layout.addWidget(tg)

        intensity_range = QGroupBox("Intensity Range")
        intensity_box = QHBoxLayout(intensity_range)

        self.imin_box = QLineEdit("500")
        self.imax_box = QLineEdit("2500")
        for b in (self.imin_box, self.imax_box):
            b.setFixedWidth(60)
            b.setDisabled(True)

        intensity_box.addWidget(QLabel("Min:"))
        intensity_box.addWidget(self.imin_box)
        intensity_box.addWidget(QLabel("Max:"))
        intensity_box.addWidget(self.imax_box)

        self.left_layout.addWidget(intensity_range)

        # Lifetime
        self.range_group = QGroupBox("Lifetime Range (ms)")
        lifetime_box = QHBoxLayout(self.range_group)

        self.vmin_box = QLineEdit("0.0")
        self.vmax_box = QLineEdit("0.1")
        for b in (self.vmin_box, self.vmax_box):
            b.setFixedWidth(60)

        lifetime_box.addWidget(QLabel("Min:"))
        lifetime_box.addWidget(self.vmin_box)
        lifetime_box.addWidget(QLabel("Max:"))
        lifetime_box.addWidget(self.vmax_box)

        self.left_layout.addWidget(self.range_group)

    def update_roi_from_boxes(self):
        try:
            xmin = int(self.roi_xmin.text())
            xmax = int(self.roi_xmax.text())
            ymin = int(self.roi_ymin.text())
            ymax = int(self.roi_ymax.text())

            xmin, xmax = sorted([xmin, xmax])
            ymin, ymax = sorted([ymin, ymax])

            self.roi = (xmin, xmax, ymin, ymax)

            self.D_set["ROI"] = self.roi

        except ValueError:
            return
        
    def update_range_mode(self):
        fit_type = self.D_set.get("fit_type", "Oxygen")

        if fit_type == "pH":
            self.range_group.setTitle("Ratio Range")
            self.vmin_box.setText("0.1")
            self.vmax_box.setText("1.0")
        else:
            self.range_group.setTitle("Lifetime Range (ms)")
            self.vmin_box.setText("0.0")
            self.vmax_box.setText("0.1")
        
        self.update_display_modes()
        
    def update_display_modes(self):
        fit_type = self.D_set.get("fit_type", "Oxygen")

        current = self.select_display.currentText()

        self.select_display.blockSignals(True)
        self.select_display.clear()

        if fit_type == "pH":
            self.select_display.addItems(
                ["All Images", "Ratio", "Gate 1", "Gate 2", "Background"]
            )
            if current == "Ratio":
                self.select_display.setCurrentText("Ratio")
            else:
                self.select_display.setCurrentIndex(0)
        else:
            self.select_display.addItems(
                ["All Images", "Lifetime", "Gate 1", "Gate 2", "Background"]
            )
            if current == "Lifetime":
                self.select_display.setCurrentText("Lifetime")
            else:
                self.select_display.setCurrentIndex(0)

        self.select_display.blockSignals(False)
    
    def select_file(self):
        """Let user choose a .h5 file manually."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Measurement File",
            self.D_set["dir_path"],
            "HDF5 Files (*.h5)"
        )
        if path:
            self.current_file = path
            self.file_label.setText(path)
        
            if "crop" in self.D_set:
                del self.D_set["crop"]
        
    def on_mode_changed(self, mode):
        if mode in ["Lifetime", "Ratio"]:
            self.vmin_box.setDisabled(False)
            self.vmax_box.setDisabled(False)
            self.imin_box.setDisabled(True)
            self.imax_box.setDisabled(True)

        elif mode in ["Gate 1", "Gate 2", "Background"]:
            self.imin_box.setDisabled(False)
            self.imax_box.setDisabled(False)
            self.vmin_box.setDisabled(True)
            self.vmax_box.setDisabled(True)

        elif mode == "All Images":
            self.vmin_box.setDisabled(False)
            self.vmax_box.setDisabled(False)
            self.imin_box.setDisabled(True)
            self.imax_box.setDisabled(True)

    def show_image_with_roi(self, img, title, cmap="viridis", vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        display_img = img
        extent = None

        if "crop" in self.D_set:
            xmin, xmax, ymin, ymax = self.D_set["crop"]
            display_img = img[ymin:ymax, xmin:xmax]
            extent = [xmin, xmax, ymax, ymin]  

        im = ax.imshow(
            display_img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=extent
        )

        plt.colorbar(im, ax=ax)
        ax.set_title(f"{title} — Select ROI")
        ax.axis("off")

        self._roi_fig = fig
        self._roi_ax = ax

        def on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return

            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            self.roi = (xmin, xmax, ymin, ymax)
            self.D_set["ROI"] = self.roi

            self.roi_xmin.setText(str(xmin))
            self.roi_xmax.setText(str(xmax))
            self.roi_ymin.setText(str(ymin))
            self.roi_ymax.setText(str(ymax))

        self.selector = RectangleSelector(
        ax,
        onselect=on_select,
        useblit=True,
        interactive=False,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        props=dict(edgecolor='cornflowerblue', facecolor='none', linewidth=1.5)
    )

        plt.show()
    
    def compute_roi_stats(self):
        fit_type = self.D_set.get("fit_type", "Oxygen")

        if fit_type == "pH":
            self.compute_ratio_roi()
        else:
            self.compute_lifetime_roi()
        
    def compute_ratio_roi(self):
        ratio, _, _ = compute_mean_ratio(
            self.current_file,
            G1_ref=None,
            G2_ref=None
        )

        xmin, xmax, ymin, ymax = self.roi
        roi_ratio = ratio[ymin:ymax, xmin:xmax]

        mean_ratio = np.nanmean(roi_ratio)
        std_ratio  = np.nanstd(roi_ratio)

        QMessageBox.information(
            self,
            "ROI Ratio",
            f"Mean ratio: {mean_ratio:.4f}\n"
            f"STD: {std_ratio:.4f}"
        )

        q, ok = QInputDialog.getDouble(
            self,
            "pH calibration",
            "pH value:",
            decimals=2
        )

        if ok:
            self.calibration_Q = np.append(self.calibration_Q, q)
            self.calibration_tau = np.append(self.calibration_tau, mean_ratio)

            if not hasattr(self, "calibration_ratio_std"):
                self.calibration_ratio_std = np.array([])

            self.calibration_ratio_std = np.append(self.calibration_ratio_std, std_ratio)

            self.label_calib_points.setText(
                f"{len(self.calibration_Q)} calibration points"
            )
    
    def show_full_resolution_figure(self):
        if not hasattr(self, "last_figure") or self.last_figure is None:
            QMessageBox.information(
                self, "No figure", "Please show a figure first."
            )
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.last_figure.savefig(tmp.name, dpi=300, bbox_inches="tight")

        img = plt.imread(tmp.name)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def compute_lifetime_roi(self):
        if self.roi is None:
            QMessageBox.warning(self, "ROI missing", "Please select an ROI first.")
            return

        if not hasattr(self, "tau_avg"):
            QMessageBox.warning(
                self,
                "No lifetime",
                "Please compute or show a lifetime image first."
            )
            return

        xmin, xmax, ymin, ymax = self.roi
        roi_tau = self.tau_avg[ymin:ymax, xmin:xmax]

        mean_tau = np.nanmean(roi_tau)
        std_tau = np.nanstd(roi_tau)

        self.last_mean = mean_tau
        self.last_std = std_tau

        QMessageBox.information(
            self,
            "ROI Lifetime",
            f"Mean τ: {mean_tau*1000:.2f} µs\n"
            f"STD τ: {std_tau*1000:.2f} µs"
        )

        if self.D_set.get("fit_type") == "Oxygen":

            q, ok = QInputDialog.getDouble(
                self,
                "Oxygen Calibration",
                "pO2 value:",
                decimals=1
            )

        else:
            q, ok = QInputDialog.getDouble(
                self,
                "Temperature Calibration",
                "Temperature value (°C):",
                decimals=1
            )
            q += 273.15

        if ok and hasattr(self, "calibration_Q"):

            self.calibration_Q = np.append(self.calibration_Q, q)
            self.calibration_tau = np.append(self.calibration_tau, mean_tau)

            if not hasattr(self, "calibration_ratio_std"):
                self.calibration_ratio_std = np.array([])

            self.calibration_ratio_std = np.append(self.calibration_ratio_std, std_tau)

            if hasattr(self, "label_calib_points"):
                self.label_calib_points.setText(
                    f"{len(self.calibration_Q)} calibration points"
                )

    def save_roi(self):
        if self.roi is None:
            QMessageBox.warning(self, "No ROI", "No ROI selected to save.")
            return

        default_dir = self.D_set["dir_path"]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI As",
            os.path.join(default_dir, "roi.json"),
            "ROI Files (*.json *.txt);;All Files (*)"
        )

        if not path:
            return 

        xmin, xmax, ymin, ymax = self.roi

        roi_data = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax
        }

        import json
        with open(path, "w") as f:
            json.dump(roi_data, f, indent=2)

        QMessageBox.information(self, "Saved", f"ROI saved to:\n{path}")

    def on_show_figure_clicked(self):
        self.allow_roi_popup = True 
        self.show_figures()

    def show_figures(self):

        filename = self.current_file
        if not os.path.exists(filename):
            QMessageBox.warning(self, "Error", "Selected file does not exist.")
            return

        params = functions.hdf5_read(filename, "parameters")
        delta_t = float(np.asarray(params["delta_t"]).item())
        exp = int(np.asarray(params["number_exposure"]).item())
        num_img = int(np.asarray(params["nbr_shots_total"]).item())
        material_raw = params["name_phosphor"].item()

        bg = functions.hdf5_read(self.current_file, "background")
        bg1 = bg["background1"]
        bg2 = bg["background2"]

        if isinstance(material_raw, bytes):
            material = material_raw.decode("utf-8")
        else:
            material = str(material_raw)

        with h5py.File(filename, "r") as f:
            groups_ratio = sorted([g for g in f.keys() if g.isdigit()])
            groups_gates = sorted([g for g in f.keys() if g.startswith("gates_")])

        groups_ratio = self.get_selected_groups(groups_ratio)
        groups_gates = self.get_selected_groups(groups_gates)

        if groups_ratio is None or groups_gates is None:
            return

        accum = None
        G1_accum = None
        G2_accum = None
        count = 0

        if self.D_set.get("fit_type") != "pH":

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
                QMessageBox.warning(self, "Error", "No valid Ratio frames found.")
                return

            tau_avg = accum / count
            self.tau_avg = tau_avg

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
            QMessageBox.warning(self, "Error", "No gate frames found.")
            return

        G1_avg = G1_accum / gate_count
        G2_avg = G2_accum / gate_count 

        mode = self.select_display.currentText()

        if mode == "All Images":

            if self.D_set.get("fit_type") == "pH":
                tau_avg = None

            try:
                thresh = float(self.thresh_box.text())
            except ValueError:
                thresh = 0
                self.thresh_box.setText("0")

            if self.D_set.get("fit_type") != "pH":
                mask = (G1_avg < thresh) | (G2_avg < thresh)

                tau_avg = tau_avg.copy()
                tau_avg[mask] = np.nan
                self.tau_avg = tau_avg

            self.show_all_images_popup(tau_avg, G1_avg, G2_avg)
            return


        elif mode == "Lifetime":

            try:
                thresh = float(self.thresh_box.text())
            except ValueError:
                thresh = 0
                self.thresh_box.setText("0")

            mask = (G1_avg < thresh) | (G2_avg < thresh)

            tau_avg = tau_avg.copy()
            tau_avg[mask] = np.nan
            self.tau_avg = tau_avg

            img = tau_avg
            cmap = "viridis"
            title = "Lifetime"

            try:
                vmin = float(self.vmin_box.text())
                vmax = float(self.vmax_box.text())
            except ValueError:
                QMessageBox.warning(self, "Error", "Enter numeric vmin/vmax.")
                return

            if vmax <= vmin:
                QMessageBox.warning(self, "Error", "vmax must be > vmin.")
                return

        elif mode == "Gate 1":
            img = G1_avg
            cmap = "viridis"
            title = "Gate 1"
            vmin = float(self.imin_box.text())
            vmax = float(self.imax_box.text())

        elif mode == "Gate 2": 
            img = G2_avg
            cmap = "viridis"
            title = "Gate 2"
            vmin = float(self.imin_box.text())
            vmax = float(self.imax_box.text())
        
        elif mode == "Background":
            img = bg1
            cmap = "viridis"
            title = "Background"
            vmin = float(self.imin_box.text())
            vmax = float(self.imax_box.text())

        elif mode == "Ratio":

            if self.D_set.get("fit_type") != "pH":
                QMessageBox.warning(self, "Invalid mode",
                                    "Ratio display is only available for pH.")
                return

            img, _, _ = compute_mean_ratio(self.current_file)

            title = "Ratio (G2 / G1)"
            cmap  = "viridis"

            try:
                vmin = float(self.vmin_box.text())
                vmax = float(self.vmax_box.text())
            except ValueError:
                vmin, vmax = 0.1, 1.0

            if vmax <= vmin:
                QMessageBox.warning(self, "Invalid range",
                                    "vmax must be greater than vmin.")
                return

        self._last_img = img
        self._last_cmap = cmap
        self._last_vmin = vmin
        self._last_vmax = vmax
        self._last_title = title

        fig = Figure(figsize=(8, 8), dpi=100)
        self.last_figure = fig
        canvas = FigureCanvasAgg(fig)

        ax = fig.add_axes([0.03, 0.03, 0.78, 0.92])

        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.axis("off")

        cax = fig.add_axes([0.84, 0.15, 0.03, 0.70])
        cbar = fig.colorbar(im, cax=cax)

        if mode == "Lifetime":
            cbar.set_label("Lifetime (ms)", fontsize=CBAR_SIZE)
        elif mode == "Gate 1":
            cbar.set_label("Intensity", fontsize=CBAR_SIZE)
        elif mode == "Gate 2":
            cbar.set_label("Intensity", fontsize=CBAR_SIZE)
        elif mode == "Ratio":
            cbar.set_label("Ratio", fontsize=CBAR_SIZE)

        cbar.ax.tick_params(labelsize=CBAR_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)

        stats = (
            # f"{material}\n"
            f"{exp} exposures\n"
            f"{num_img} images\n"
            f"Δt = {delta_t*1e6:.1f} µs"
        )

        if hasattr(self, "last_mean"):
            stats += (
                f"\nMean τ = {self.last_mean*1000:.2f} µs"
                f"\nSTD τ = {self.last_std*1000:.2f} µs"
            )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)

        pix = QPixmap()
        pix.loadFromData(buf.getvalue())

        self.last_pixmap = pix
        self.fig_label.setPixmap(pix)

        self.last_figure = fig

        if mode == "Lifetime" and self.allow_roi_popup:
            self.show_image_with_roi(tau_avg, "Lifetime", vmin=vmin, vmax=vmax)

        elif mode == "Ratio" and self.allow_roi_popup:
            self.show_image_with_roi(img, "Ratio", vmin=vmin, vmax=vmax)

    def show_all_images_popup(self, tau_avg, G1_avg, G2_avg):

        cmap="viridis"

        if self.D_set.get("fit_type") == "pH":

            ratio, G1_mean, G2_mean = compute_mean_ratio(self.current_file)

            try:
                rmin = float(self.vmin_box.text())
                rmax = float(self.vmax_box.text())
            except ValueError:
                rmin, rmax = 0.0, 1.0

            try:
                thresh = float(self.thresh_box.text())
            except ValueError:
                thresh = 0
                self.thresh_box.setText("0")

            mask = (G1_mean < thresh) | (G2_mean < thresh)

            ratio_plot = ratio.copy()
            ratio_plot[mask] = np.nan

            G1_plot = G1_mean.copy()
            G2_plot = G2_mean.copy()

            G1_plot[mask] = np.nan
            G2_plot[mask] = np.nan

            g1_min = np.nanpercentile(G1_plot, 1)
            g2_min = np.nanpercentile(G2_plot, 1)
            g1_max = np.nanpercentile(G1_plot, 99)
            g2_max = np.nanpercentile(G2_plot, 99)

            images = [
                (ratio_plot,   "Ratio (G2 / G1)", rmin, rmax),
                (G1_plot, "G1 mean", g1_min, g1_max),
                (G2_plot, "G2 mean", g2_min, g2_max),
            ]


            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.canvas.manager.set_window_title("All Images — Ratio, Gate 1, Gate 2")

            for ax, (img, title, vmin, vmax) in zip(axes, images):
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=TITLE_SIZE)
                ax.axis("off")

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=12)

                if "Ratio" in title:
                    label = "Ratio (G2 / G1)"
                else:
                    label = "Intensity"

                cbar.set_label("")
                cbar.ax.text(
                    0.5, 1.03,
                    label,
                    ha="center", va="bottom",
                    fontsize=CBAR_SIZE,
                    transform=cbar.ax.transAxes
                )

            fig.tight_layout()
            plt.show()
            return

        try:
            vmin_live = float(self.vmin_box.text())
            vmax_live = float(self.vmax_box.text())
        except ValueError:
            vmin_live, vmax_live = np.nanmin(tau_avg), np.nanmax(tau_avg)

        vmin_g1 = np.nanpercentile(G1_avg, 1)
        vmax_g1 = np.nanpercentile(G1_avg, 99)
        vmin_g2 = np.nanpercentile(G2_avg, 1)
        vmax_g2 = np.nanpercentile(G2_avg, 99)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.canvas.manager.set_window_title("All Images — Lifetime, Gate 1, Gate 2")

        # lifetime
        ax = axes[0]
        im = ax.imshow(tau_avg, cmap="inferno", vmin=vmin_live, vmax=vmax_live)
        ax.set_title("Lifetime (ms)", fontsize=TITLE_SIZE)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("")

        cbar.ax.tick_params(labelsize=12)

        cbar.ax.text(
            0.5, 1.03,                     
            "Lifetime (ms)",               
            ha='center', va='bottom',
            fontsize=CBAR_SIZE,
            transform=cbar.ax.transAxes
        )

        # gate 1
        ax = axes[1]
        im1 = ax.imshow(G1_avg, cmap=cmap, vmin=vmin_g1, vmax=vmax_g1)
        ax.set_title("Gate 1", fontsize=TITLE_SIZE)
        ax.axis("off")
        cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("")

        cbar1.ax.tick_params(labelsize=12)

        cbar1.ax.text(
            0.5, 1.03,
            "Intensity",
            ha='center', va='bottom',
            fontsize=CBAR_SIZE,
            transform=cbar1.ax.transAxes
        )

        # gate 2
        ax = axes[2]
        im2 = ax.imshow(G2_avg, cmap=cmap, vmin=vmin_g2, vmax=vmax_g2)
        ax.set_title("Gate 2", fontsize=TITLE_SIZE)
        ax.axis("off")
        cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label("")

        cbar2.ax.tick_params(labelsize=12)

        cbar2.ax.text(
            0.5, 1.03,
            "Intensity",
            ha='center', va='bottom',
            fontsize=CBAR_SIZE,
            transform=cbar2.ax.transAxes
        )

        fig.tight_layout()
        plt.show()

    def save_figure(self):
        if not hasattr(self, "last_figure") or self.last_figure is None:
            QMessageBox.warning(self, "No Figure", "No figure is available to save.")
            return
        
        for ax in self.last_figure.axes:
            ax.title.set_fontsize(18)
            ax.xaxis.label.set_fontsize(16)
            ax.yaxis.label.set_fontsize(16)
            ax.tick_params(labelsize=14)

        for ax in self.last_figure.axes:
            if hasattr(ax, "_colorbar"):
                ax._colorbar.ax.tick_params(labelsize=14)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            self.D_set["dir_path"],
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)"
        )

        if not path:
            return

        self.last_figure.savefig(path, dpi=600, bbox_inches="tight")
        QMessageBox.information(self, "Saved", f"Figure saved:\n{path}")

    def get_selected_groups(self, groups):

        if not hasattr(self, "image_number_box"):
            return groups

        text = self.image_number_box.text().strip()

        if text == "":
            return groups

        try:
            idx = int(text) - 1

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid input",
                "Please enter a valid image number."
            )
            return None

        if idx < 0 or idx >= len(groups):
            QMessageBox.warning(
                self,
                "Out of range",
                f"Image number must be between 1 and {len(groups)}."
            )
            return None

        return [groups[idx]]

