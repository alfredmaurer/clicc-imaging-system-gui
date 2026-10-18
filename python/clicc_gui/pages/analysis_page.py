import os
from io import BytesIO
import cv2
import pandas as pd

import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from PyQt6.QtWidgets import (
    QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QHBoxLayout,
    QGroupBox, QFileDialog, QInputDialog,
)
from PyQt6.QtGui import QPixmap, QAction, QIcon
from PyQt6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.widgets import RectangleSelector
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

import functions
import fitting_models

from analysis_base import BaseImageAnalysisPage
from functions import compute_mean_ratio
from globals_config import D_set
from fitting_models import pH_from_ratio_boltz, pO2_from_tau, pO2_from_tau_partial
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TITLE_SIZE = 16
CBAR_SIZE = 14
LABEL_SIZE = 14


class AnalysisGUI(BaseImageAnalysisPage):
    def __init__(self, D_set):
        super().__init__(D_set, title="Analysis Controls")

        self.profile_rois = []   

        self.btn_set_crop = QPushButton("Set crop")
        self.btn_set_crop.clicked.connect(self.set_crop)

        btn_calc = QPushButton("Calculate")
        btn_calc.clicked.connect(self.calculate)

        self.btn_save_fig = QPushButton("Save Figure")
        self.btn_save_fig.clicked.connect(self.save_figure)

        self.btn_add_profile_roi = QPushButton("Add profile rectangle")
        self.btn_add_profile_roi.clicked.connect(self.add_profile_rectangle)

        self.btn_clear_profile_rois = QPushButton("Clear profile rectangles")
        self.btn_clear_profile_rois.clicked.connect(self.clear_profile_rectangles)

        self.btn_plot_profiles = QPushButton("Plot pH profile(s)")
        self.btn_plot_profiles.clicked.connect(self.create_pH_profiles_multi)

        self.profile_group = QGroupBox("pH Profiles")
        profile_layout = QVBoxLayout(self.profile_group)

        profile_layout.addWidget(self.btn_add_profile_roi)
        profile_layout.addWidget(self.btn_clear_profile_rois)
        profile_layout.addWidget(self.btn_plot_profiles)

        self.btn_video = QPushButton("Create Video")
        self.btn_video.clicked.connect(self.create_video)

        self.image_select_group = QGroupBox("Image Selection")
        img_layout = QHBoxLayout(self.image_select_group)

        img_layout.addWidget(QLabel("Image Number:"))

        self.image_number_box = QLineEdit()
        self.image_number_box.setPlaceholderText("all")
        self.image_number_box.setFixedWidth(70)

        hint = (
            "Leave empty to average all images.\n"
            "Enter an image number to analyze only one image\n"
            "from the measurement series."
        )

        action = QAction(
            QIcon(str(BASE_DIR / "exclamation_mark.png")),
            "",
            self.image_number_box
        )

        action.setToolTip(hint)

        self.image_number_box.addAction(
            action,
            QLineEdit.ActionPosition.TrailingPosition
        )

        img_layout.addWidget(self.image_number_box)

        self.page_controls.addWidget(self.image_select_group)

        self.scale_group = QGroupBox("Spatial scale")
        scale_layout = QHBoxLayout(self.scale_group)

        self.btn_set_scale = QPushButton("Set scale (mm)")
        self.btn_set_scale.clicked.connect(self.set_spatial_scale)

        self.scale_edit = QLineEdit()
        self.scale_edit.setFixedWidth(90)
        self.scale_edit.setPlaceholderText("px / mm")
        self.scale_edit.editingFinished.connect(self.on_scale_edited)

        scale_layout.addWidget(self.btn_set_scale)
        scale_layout.addWidget(QLabel("px/mm:"))
        scale_layout.addWidget(self.scale_edit)

        self.analysis_range_group = QGroupBox("Analysis Value Range")
        ar_layout = QHBoxLayout(self.analysis_range_group)

        self.amin_box = QLineEdit("0.0")
        self.amax_box = QLineEdit("100.0")

        for b in (self.amin_box, self.amax_box):
            b.setFixedWidth(60)

        ar_layout.addWidget(QLabel("Min:"))
        ar_layout.addWidget(self.amin_box)
        ar_layout.addWidget(QLabel("Max:"))
        ar_layout.addWidget(self.amax_box)

        self.page_controls.addWidget(self.scale_group)
        self.page_controls.addWidget(self.btn_set_crop)
        self.page_controls.addWidget(btn_calc)
        self.page_controls.addWidget(self.btn_save_fig)

        self.page_controls.addWidget(self.profile_group)

        self.page_controls.addWidget(self.btn_video)
        self.page_controls.addWidget(self.analysis_range_group)
        self.update_analysis_mode_visibility()
        self.set_default_analysis_range()

    
    def update_analysis_mode_visibility(self):
        fit_type = str(self.D_set.get("fit_type", "")).strip().lower()
        is_pH = (fit_type == "ph")

        for attr in (
            "btn_set_crop",
            "scale_group",
        ):
            getattr(self, attr).setVisible(True)

        if hasattr(self, "profile_group"):
            self.profile_group.setVisible(is_pH)

        if hasattr(self, "btn_single_gate"):
            self.btn_single_gate.setVisible(not is_pH)

        if hasattr(self, "btn_video"):
            self.btn_video.setVisible(True)

    def calculate(self):
        analysis_type = self.D_set.get("fit_type", "Oxygen")

        if analysis_type == "pH":

            ratio, G1_mean, G2_mean = compute_mean_ratio(
                self.current_file
            )

            thresh = float(self.thresh_box.text())
            valid_intensity = (G1_mean > thresh) & (G2_mean > thresh)

            self.G1_corr = G1_mean
            self.G2_corr = G2_mean
            self.last_ratio_map = ratio

            required = ["Top", "Bottom", "pKa", "slope"]
            for k in required:
                if k not in self.D_set:
                    QMessageBox.warning(
                        self, "Missing calibration",
                        "Please run pH fitting first."
                    )
                    return

            pH_map = np.full_like(ratio, np.nan)

            pH_map[valid_intensity] = fitting_models.pH_from_ratio_boltz(
                ratio[valid_intensity],
                self.D_set["Top"],
                self.D_set["Bottom"],
                self.D_set["pKa"],
                self.D_set["slope"]
            )

            pH_map[~np.isfinite(pH_map)] = np.nan

            self.last_crop  = self.D_set.get("crop", None)

            self.last_pH_map = pH_map
            pH_mean = pH_std = None

            if self.roi is not None:
                xmin, xmax, ymin, ymax = self.roi

                roi_ratio = ratio[ymin:ymax, xmin:xmax]

                low  = min(self.D_set["Top"], self.D_set["Bottom"])
                high = max(self.D_set["Top"], self.D_set["Bottom"])

                valid = (
                    np.isfinite(roi_ratio) &
                    (roi_ratio > low) &
                    (roi_ratio < high)
                )

                pH_roi = pH_from_ratio_boltz(
                    roi_ratio[valid],
                    self.D_set["Top"],
                    self.D_set["Bottom"],
                    self.D_set["pKa"],
                    self.D_set["slope"]
                )

                pH_mean = np.nanmean(pH_roi)
                pH_std  = np.nanstd(pH_roi)

                QMessageBox.information(
                    self,
                    "ROI pH",
                    f"Mean pH: {pH_mean:.2f}\n"
                    f"STD pH: {pH_std:.2f}"
                )

            fig = Figure(figsize=(7, 6))
            canvas = FigureCanvasAgg(fig)
            ax_img = fig.add_subplot(111)

            try:
                vmin = float(self.amin_box.text())
                vmax = float(self.amax_box.text())
                if vmax <= vmin:
                    raise ValueError
            except ValueError:
                vmin, vmax = 6.0, 13.0

            display_map = pH_map
            extent = None

            xlabel = "x (px)"
            ylabel = "y (px)"

            if "crop" in self.D_set:
                xmin, xmax, ymin, ymax = self.D_set["crop"]
                display_map = pH_map[ymin:ymax, xmin:xmax]

            if "mm_per_pixel" in self.D_set:
                mm = self.D_set["mm_per_pixel"]

                h, w = display_map.shape

                extent = [
                    0, w * mm,
                    h * mm, 0
                ]

                xlabel = "x (mm)"
                ylabel = "y (mm)"
            
            self.last_pH_extent = extent    # [xmin_mm, xmax_mm, ymax_mm, ymin_mm]
            self.last_pH_shape  = display_map.shape   

            im = ax_img.imshow(
                display_map,
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
                origin="upper",
                extent=extent
            )

            self.draw_profile_rois_on_axis(ax_img)

            ax_img.set_xlabel(xlabel, fontsize=LABEL_SIZE)
            ax_img.set_ylabel(ylabel, fontsize=LABEL_SIZE)

            from matplotlib.ticker import AutoMinorLocator

            if "mm_per_pixel" in self.D_set:
                ax_img.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax_img.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax_img.tick_params(which="minor", length=3, color="gray")

            ax_img.tick_params(which="major", length=6)

            fname = os.path.basename(self.current_file)
            title = os.path.splitext(fname)[0]
            ax_img.set_title(title, fontsize=TITLE_SIZE)

            cbar = fig.colorbar(im, ax=ax_img)
            cbar.set_label("pH", fontsize = CBAR_SIZE)

            canvas.draw()
            buf = BytesIO()
            fig.savefig(buf, dpi=120, bbox_inches="tight")
            buf.seek(0)

            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            self.fig_label.setPixmap(
                pix.scaled(
                    self.fig_label.width(),
                    self.fig_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

            self.last_figure = fig

            if getattr(self, "allow_roi_popup", True):
                self.show_image_popup(
                    display_map,    
                    "pH",
                    cmap="inferno",
                    vmin=vmin,
                    vmax=vmax
                )
                            
            return

        # Lifetime based analysis (Oxygen / Temperature)
        if not hasattr(self, "tau_avg"):
            QMessageBox.warning(
                self, "No lifetime",
                "Please compute / show a lifetime image first."
            )
            return

        tau = self.tau_avg

        # Temperature Analysis
        if analysis_type == "Temperature":
            required = ["k0", "k1", "deltaE"]
            for k in required:
                if k not in self.D_set:
                    QMessageBox.warning(
                        self, "Missing calibration",
                        "Temperature calibration not available.\n"
                        "Please run Arrhenius fitting first."
                    )
                    return

            from fitting_models import T_from_tau_arrhenius

            T_K = T_from_tau_arrhenius(
                tau,
                self.D_set["k0"],
                self.D_set["k1"],
                self.D_set["deltaE"]
            )

            display_unit = self.D_set.get("temperature_unit", "°C")
            if display_unit == "°C":
                value_map = T_K - 273.15
                unit_str = "°C"
            else:
                value_map = T_K
                unit_str = "K"

            title = f"Temperature map"
            cbar_label = f"Temperature ({unit_str})"
            cmap = "inferno"
            vmin = 0
            vmin, vmax = self.get_analysis_range(value_map)

        # Oxygen Analysis
        elif analysis_type == "Oxygen":
            required = ["tau0", "Ksv1", "f", "sv_partial"]
            for k in required:
                if k not in self.D_set:
                    QMessageBox.warning(
                        self, "Missing fit",
                        "No oxygen calibration available.\n"
                        "Please run fitting first."
                    )
                    return

            if self.D_set["sv_partial"]:
                value_map = pO2_from_tau_partial(
                    tau,
                    self.D_set["tau0"],
                    self.D_set["Ksv1"],
                    self.D_set["f"]
                )
            else:
                value_map = pO2_from_tau(
                    tau,
                    self.D_set["tau0"],
                    self.D_set["Ksv1"],
                    self.D_set["Ksv2"],
                    self.D_set["f"]
                )

            unit = self.D_set.get("oxygen_unit", "%")
            if unit == "%":
                unit_str = "%"
                physical_max = 100.0
            elif unit == "kPa":
                unit_str = "kPa"
                physical_max = None
            elif unit == "mmol/L":
                unit_str = "mmol/L"
                physical_max = None
            else:
                unit_str = ""

            cmap = "inferno"
            vmin, vmax = self.get_analysis_range(value_map, physical_max)
            title = f"Oxygen map"
            cbar_label = f"Oxygen ({unit_str})"

        else:
            QMessageBox.information(
                self, "Not implemented",
                f"{analysis_type} analysis is not implemented."
            )
            return

        mean_val = std_val = None
        if self.roi is not None:
            xmin, xmax, ymin, ymax = self.roi
            roi_vals = value_map[ymin:ymax, xmin:xmax]
            mean_val = np.nanmean(roi_vals)
            std_val = np.nanstd(roi_vals)

            QMessageBox.information(
                self,
                f"ROI {analysis_type}",
                f"Mean: {mean_val:.2f} {unit_str}\n"
                f"STD: {std_val:.2f} {unit_str}"
            )
        
        display_map = value_map

        if "crop" in self.D_set:
            xmin, xmax, ymin, ymax = self.D_set["crop"]
            display_map = value_map[ymin:ymax, xmin:xmax]

        value_plot = np.ma.masked_invalid(display_map)

        fig = Figure(figsize=(7, 6), constrained_layout=True)
        canvas = FigureCanvasAgg(fig)
        ax_img = fig.add_subplot(111)

        extent = None

        xlabel = "x (px)"
        ylabel = "y (px)"

        if "mm_per_pixel" in self.D_set:

            mm = self.D_set["mm_per_pixel"]

            h, w = value_plot.shape

            extent = [
                0,
                w * mm,
                h * mm,
                0
            ]

            xlabel = "x (mm)"
            ylabel = "y (mm)"

        im = ax_img.imshow(
            value_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=extent
        )

        ax_img.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax_img.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax_img.set_title(title, fontsize=TITLE_SIZE)
        ax_img.tick_params(axis="both", labelsize=12)

        cbar = fig.colorbar(im, ax=ax_img)
        cbar.set_label(cbar_label, fontsize=CBAR_SIZE)

        canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, dpi=120, bbox_inches="tight")
        buf.seek(0)

        pix = QPixmap()
        pix.loadFromData(buf.getvalue())
        self.fig_label.setPixmap(
            pix.scaled(
                self.fig_label.width(),
                self.fig_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )
        self.last_figure = fig

        if getattr(self, "allow_roi_popup", True):
            self.show_image_popup(
                display_map,
                analysis_type,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

    def show_image_popup(self, img, title, cmap="viridis", vmin=None, vmax=None):

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper"
        )

        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.axis("off")

        plt.show()
    
    def add_profile_rectangle(self):
        if not hasattr(self, "last_pH_map") or self.last_pH_map is None:
            QMessageBox.warning(self, "No data", "Calculate pH map first.")
            return

        if "mm_per_pixel" not in self.D_set:
            QMessageBox.warning(
                self,
                "Scale not set",
                "Set mm per pixel first."
            )
            return

        pH_map = self.last_pH_map

        fig, ax = plt.subplots(figsize=(7, 6))
        display_map = pH_map

        if "crop" in self.D_set:
            xmin, xmax, ymin, ymax = self.D_set["crop"]
            display_map = pH_map[ymin:ymax, xmin:xmax]

        try:
            vmin = float(self.amin_box.text())
            vmax = float(self.amax_box.text())
            if vmax <= vmin:
                raise ValueError
        except ValueError:
            vmin, vmax = 6.0, 13.0

        extent = None
        if "crop" in self.D_set:
            cxmin, cxmax, cymin, cymax = self.D_set["crop"]
            width_px = cxmax - cxmin
            height_px = cymax - cymin
            extent = [
                0,
                width_px * self.D_set["mm_per_pixel"],
                height_px * self.D_set["mm_per_pixel"],
                0
            ]

        ax.imshow(
            display_map,
            cmap="inferno",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            extent=extent
        )

        self.draw_profile_rois_on_axis(ax)
        ax.set_title("Draw rectangle ROI for profile")

        def on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return

            mm_per_px = self.D_set["mm_per_pixel"]

            x1 = int(eclick.xdata / mm_per_px)
            y1 = int(eclick.ydata / mm_per_px)
            x2 = int(erelease.xdata / mm_per_px)
            y2 = int(erelease.ydata / mm_per_px)

            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            if "crop" in self.D_set:
                cxmin, cxmax, cymin, cymax = self.D_set["crop"]
                xmin += cxmin
                xmax += cxmin
                ymin += cymin
                ymax += cymin

            roi = (xmin, xmax, ymin, ymax)

            self.profile_rois.append(roi)

            QMessageBox.information(
                self,
                "ROI added",
                f"Added ROI #{len(self.profile_rois)}:\n"
                f"x: {xmin}–{xmax}\n"
                f"y: {ymin}–{ymax}"
            )

            plt.close(fig)

        self._profile_selector = RectangleSelector(
            ax,
            onselect=on_select,
            useblit=True,
            interactive=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            props=dict(
                edgecolor='cornflowerblue',
                facecolor='none',
                linewidth=1.5
            )
        )

        plt.show()
    
    def clear_profile_rectangles(self):
        self.profile_rois = []
        QMessageBox.information(self, "Cleared", "All stored profile ROIs were cleared.")

    def plot_profiles(self):
        if hasattr(self, "profile_rois") and len(self.profile_rois) > 0:
            return self.create_pH_profiles_multi()

        QMessageBox.warning(
            self,
            "No ROI",
            "No profile ROIs stored.\n"
            "Either add a profile rectangle or draw a normal ROI first."
        )
    
    def create_pH_profiles_multi(self):
        from scipy.ndimage import gaussian_filter1d

        if not hasattr(self, "last_pH_map") or self.last_pH_map is None:
            QMessageBox.warning(self, "No data", "Calculate pH map first.")
            return

        if "mm_per_pixel" not in self.D_set or self.D_set["mm_per_pixel"] is None:
            QMessageBox.warning(self, "Scale not set", "Set px/mm first.")
            return

        if len(self.profile_rois) == 0:
            QMessageBox.warning(self, "No ROIs", "Add at least one profile rectangle.")
            return

        pH_map = self.last_pH_map
        mm_per_px = self.D_set["mm_per_pixel"]

        low  = self.D_set.get("pH_min", 0)
        high = self.D_set.get("pH_max", 14)

        # Plot pH map with ROIs
        fig_map, ax_map = plt.subplots(figsize=(6, 6))

        display_map = pH_map
        extent = None

        if "crop" in self.D_set:
            cxmin, cxmax, cymin, cymax = self.D_set["crop"]
            display_map = pH_map[cymin:cymax, cxmin:cxmax]

            width_px  = cxmax - cxmin
            height_px = cymax - cymin

            extent = [0, width_px * mm_per_px, height_px * mm_per_px, 0]
        
        try:
            vmin = float(self.amin_box.text())
            vmax = float(self.amax_box.text())
            if vmax <= vmin:
                raise ValueError
        except ValueError:
            vmin, vmax = 6.0, 13.0

        im = ax_map.imshow(
            display_map,
            cmap="inferno",
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax
        )

        ax_map.set_title("pH map with profile ROI(s)", fontsize=18)
        ax_map.set_xlabel("x (mm)", fontsize=16)
        ax_map.set_ylabel("y (mm)", fontsize=16)

        ax_map.xaxis.set_minor_locator(MultipleLocator(1))
        ax_map.yaxis.set_minor_locator(MultipleLocator(1))

        ax_map.tick_params(which="minor", length=3, color="gray", labelsize=12)
        ax_map.tick_params(which="major", length=6, labelsize=14)

        self.draw_profile_rois_on_axis(ax_map)

        cbar = fig_map.colorbar(im, ax=ax_map)
        cbar.set_label("pH", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        fig_map.tight_layout()

        # Plot profiles
        fig, ax = plt.subplots(figsize=(8, 5))

        all_profiles = {}

        for i, roi in enumerate(self.profile_rois):
            xmin, xmax, ymin, ymax = roi
            roi_data = pH_map[ymin:ymax, xmin:xmax]

            if roi_data.size == 0:
                continue

            roi_data = roi_data.copy()
            roi_data[(roi_data < low) | (roi_data > high)] = np.nan

            h, w = roi_data.shape

            if h >= w:
                mean_profile = np.nanmean(roi_data, axis=1)
                std_profile  = np.nanstd(roi_data, axis=1)
                axis_len = h
                direction = "vertical"
            else:
                mean_profile = np.nanmean(roi_data, axis=0)
                std_profile  = np.nanstd(roi_data, axis=0)
                axis_len = w
                direction = "horizontal"

            mean_profile = gaussian_filter1d(mean_profile, sigma=2)

            x_mm = np.linspace(0, axis_len * mm_per_px, axis_len)

            pos_text = "pos unknown"

            if hasattr(self, "last_pH_extent") and hasattr(self, "last_pH_shape") and "crop" in self.D_set:

                xmin_c, xmax_c, ymin_c, ymax_c = self.D_set["crop"]
                xmin_mm, xmax_mm, ymax_mm, ymin_mm = self.last_pH_extent
                H, W = self.last_pH_shape

                rx0 = xmin - xmin_c
                rx1 = xmax - xmin_c
                ry0 = ymin - ymin_c
                ry1 = ymax - ymin_c

                mm_per_px_x = (xmax_mm - xmin_mm) / W
                mm_per_px_y = (ymax_mm - ymin_mm) / H

                if direction == "vertical":
                    x0 = xmin_mm + rx0 * mm_per_px_x
                    x1 = xmin_mm + rx1 * mm_per_px_x
                    pos_text = f"x={x0:.1f}-{x1:.1f} mm"
                else:
                    y0 = ymin_mm + ry0 * mm_per_px_y
                    y1 = ymin_mm + ry1 * mm_per_px_y
                    pos_text = f"y={y0:.1f}-{y1:.1f} mm"

            label_txt = f"Profile {i+1} ({pos_text})"

            df = pd.DataFrame({
                "depth": x_mm,
                "mean_pH": mean_profile,
                "std_pH": std_profile
            })

            sheet_name = f"Profile_{i+1}"
            all_profiles[sheet_name] = df

            ax.plot(x_mm, mean_profile, lw=2, label=label_txt)

            ax.fill_between(
                x_mm,
                mean_profile - std_profile,
                mean_profile + std_profile,
                alpha=0.2
            )

        ax.set_xlabel(
            "Depth (mm)" if direction == "vertical" else "Position (mm)",
            fontsize=18
        )
        ax.set_ylabel("pH", fontsize=16)
        ax.set_title("pH Profile(s)", fontsize=18)
        
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(which="minor", length=3, color="gray", labelsize=12)
        ax.tick_params(which="major", length=6, labelsize=14)

        ax.legend(fontsize=14)

        fig.tight_layout()

        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        default_path = os.path.join(self.D_set["dir_path"], base_name + ".xlsx")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save profiles to Excel",
            default_path,
            "Excel file (*.xlsx)"
        )

        if not save_path:
            QMessageBox.information(self, "Cancelled", "Saving cancelled.")
            return

        if not save_path.lower().endswith(".xlsx"):
            save_path += ".xlsx"

        folder = os.path.dirname(save_path)
        base   = os.path.splitext(os.path.basename(save_path))[0]

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            for sheet, df in all_profiles.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)

        fig_map_path = os.path.join(folder, base + "_profile_roi.png")
        fig_path     = os.path.join(folder, base + "_profile.png")

        fig_map.savefig(fig_map_path, dpi=300, bbox_inches="tight")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

        QMessageBox.information(
            self,
            "Saved",
            f"Saved:\n\n"
            f"{save_path}\n"
            f"{fig_map_path}\n"
            f"{fig_path}"
        )

        plt.show()


    def draw_profile_rois_on_axis(self, ax):

        if len(self.profile_rois) == 0:
            return

        if "crop" in self.D_set:
            cxmin, cxmax, cymin, cymax = self.D_set["crop"]
        else:
            cxmin = cymin = 0
        mm = self.D_set.get("mm_per_pixel", 1.0)

        for i, roi in enumerate(self.profile_rois):
            xmin, xmax, ymin, ymax = roi

            x0 = (xmin - cxmin) * mm
            y0 = (ymin - cymin) * mm
            w  = (xmax - xmin) * mm
            h  = (ymax - ymin) * mm

            rect = Rectangle(
                (x0, y0),
                w, h,
                linewidth=2,
                edgecolor="dodgerblue",
                facecolor="none",
                alpha=0.6,
                zorder=10
            )
            ax.add_patch(rect)

    def invalidate_pH_map(self):
        self.last_pH_map = None

    def set_spatial_scale(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF with mm reference",
            self.D_set["dir_path"],
            "TIFF images (*.tif *.tiff)"
        )
        if not path:
            return

        img = imageio.imread(path)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img, cmap="gray")
        ax.set_title("Drag a line over the mm reference")
        ax.axis("off")

        line_artist = ax.plot([], [], color="red", lw=2)[0]

        def on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return

            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            # draw line
            line_artist.set_data([x1, x2], [y1, y2])
            fig.canvas.draw_idle()

            pixel_dist = np.hypot(x2 - x1, y2 - y1)

            if pixel_dist <= 1:
                QMessageBox.warning(self, "Invalid selection", "Line too short.")
                return

            mm, ok = QInputDialog.getDouble(
                self,
                "Reference length",
                "Length of drawn line (mm):",
                decimals=3,
                min=0.0001
            )
            if not ok:
                return

            mm_per_pixel = mm / pixel_dist
            self.D_set["mm_per_pixel"] = mm_per_pixel
            if hasattr(self, "invalidate_pH_map"):
                self.invalidate_pH_map()

            px_per_mm = 1.0 / mm_per_pixel
            self.scale_edit.setText(f"{px_per_mm:.2f}")

            QMessageBox.information(
                self,
                "Spatial scale set",
                f"Scale:\n{mm_per_pixel:.6f} mm / pixel\n"
                f"({px_per_mm:.2f} px / mm)"
            )

            plt.close(fig)

        self._scale_selector = RectangleSelector(
            ax,
            onselect=on_select,
            useblit=True,
            interactive=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            props=dict(edgecolor="red", linewidth=2, fill=False)
        )

        plt.show(block=False)
        self._scale_changed = True
        self.last_pH_map = None
        self.last_pH_extent = None


    def on_scale_edited(self):
        try:
            px_per_mm = float(self.scale_edit.text())
            if px_per_mm <= 0:
                raise ValueError

            self.D_set["mm_per_pixel"] = 1.0 / px_per_mm
            if hasattr(self, "invalidate_pH_map"):
                self.invalidate_pH_map()

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid scale",
                "Please enter a positive number (pixels per mm)."
            )
            self.scale_edit.setText("")
            self._scale_changed = True
            self.last_pH_map = None
            self.last_pH_extent = None

    def set_crop(self):
        try:
            with h5py.File(self.current_file, "r") as f:
                groups = sorted(
                    g for g in f.keys()
                    if g.startswith("gates_")
                )

                if len(groups) == 0:
                    raise ValueError("No gate images found.")

                G1_mean = f[groups[0]]["G1"][:]

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(G1_mean, cmap="gray")
        ax.set_title("Drag to select crop region")
        ax.axis("off")

        self._crop_selector = None  

        def on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return

            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            self.crop = (xmin, xmax, ymin, ymax)
            self.D_set["crop"] = self.crop

            self.roi = None
            self.D_set.pop("ROI", None)

            if hasattr(self, "invalidate_pH_map"):
                self.invalidate_pH_map()

            QMessageBox.information(
                self,
                "Crop set",
                f"x: {xmin} – {xmax}\n"
                f"y: {ymin} – {ymax}"
            )

            plt.close(fig)

        self._crop_selector = RectangleSelector(
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

        plt.show(block=False)

    def calculate_single_gate_lifetime_multifile(self):
        import h5py
        from scipy.optimize import curve_fit

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select measurements with different gate widths",
            self.D_set["dir_path"],
            "HDF5 files (*.h5)"
        )

        if len(files) < 3:
            QMessageBox.warning(
                self,
                "Not enough data",
                "Please select at least 3 measurements\n"
                "with different gate widths."
            )
            return

        if self.roi is None:
            QMessageBox.warning(self, "ROI missing", "Please define an ROI first.")
            return

        xmin, xmax, ymin, ymax = self.roi

        gate_widths = []
        intensities = []

        for file in files:
            params = functions.hdf5_read(file, "parameters")
            delta_t = float(np.asarray(params["delta_t"]).item()) 

            bg = functions.hdf5_read(file, "background")
            bg1 = bg["background1"]

            with h5py.File(file, "r") as f:
                gates = sorted(g for g in f.keys() if g.startswith("gates_"))
                if not gates:
                    continue
                g = gates[0]

            data = functions.hdf5_read(file, g)
            G1 = data["G1"].astype(float) - bg1

            roi_img = G1[ymin:ymax, xmin:xmax]
            I = np.nanmean(roi_img)

            if np.isfinite(I) and I > 0:
                gate_widths.append(delta_t)
                intensities.append(I)

        gate_widths = np.asarray(gate_widths)
        intensities = np.asarray(intensities)

        if len(gate_widths) < 3:
            QMessageBox.critical(
                self,
                "Invalid data",
                "Could not extract valid intensity data."
            )
            return

        idx = np.argsort(gate_widths)
        gate_widths = gate_widths[idx]
        intensities = intensities[idx]

        def gate_model(dt, A, tau):
            return A * (1.0 - np.exp(-dt / tau))

        p0 = (np.max(intensities), np.mean(gate_widths))

        params, cov = curve_fit(
            gate_model,
            gate_widths,
            intensities,
            p0=p0,
            maxfev=10000
        )

        A_fit, tau_fit = params
        tau_std = np.sqrt(cov[1, 1])

        fig = Figure(figsize=(6, 5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.scatter(
            gate_widths * 1e6,
            intensities,
            color="tab:blue",
            label="Measurements"
        )

        t_fit = np.linspace(gate_widths.min(), gate_widths.max(), 300)
        ax.plot(
            t_fit * 1e6,
            gate_model(t_fit, A_fit, tau_fit),
            color="red",
            label="Fit"
        )

        ax.set_xlabel("Gate width (µs)")
        ax.set_ylabel("ROI mean intensity")
        ax.set_title(f"Single-gate lifetime τ = {tau_fit*1e3:.3f} µs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        canvas.draw()

        buf = BytesIO()
        fig.savefig(buf, dpi=120, bbox_inches="tight")
        buf.seek(0)

        pix = QPixmap()
        pix.loadFromData(buf.getvalue())

        self.fig_label.setPixmap(
            pix.scaled(
                self.fig_label.width(),
                self.fig_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )
        self.last_figure = fig

        QMessageBox.information(
            self,
            "Single-gate lifetime result",
            f"τ = {tau_fit*1e3:.3f} ± {tau_std*1e3:.3f} µs\n"
            f"({len(gate_widths)} measurements)"
        )

    def create_video(self):
        h5_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select measurement file",
            self.D_set["dir_path"],
            "HDF5 files (*.h5)"
        )
        if not h5_path:
            return

        video_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save video as",
            self.D_set["dir_path"],
            "MP4 video (*.mp4)"
        )
        if not video_path:
            return

        try:
            self.generate_analysis_video(h5_path, video_path)
            QMessageBox.information(self, "Done", f"Video saved to:\n{video_path}")
        except Exception as e:
            QMessageBox.critical(self, "Video error", str(e))

    def generate_analysis_video(self, h5_file, video_name):

        roi = getattr(self, "roi", None)

        def apply_crop(img):
            return img
        
        def apply_roi(img):
            if roi is None:
                return img, None

            xmin, xmax, ymin, ymax = roi

            mask = np.zeros_like(img, dtype=bool)
            mask[ymin:ymax, xmin:xmax] = True

            img_roi = img.copy()
            img_roi[~mask] = np.nan

            return img_roi, (xmin, xmax, ymin, ymax)

        fit_type = self.D_set["fit_type"]

        params = functions.hdf5_read(h5_file, "parameters")
        fps = float(np.asarray(params["fps"]).item())
        t1 = float(np.asarray(params["delta_t"]).item())

        if fit_type == "Oxygen":

            tau0 = self.D_set["tau0"] * 1e-3
            Ksv1 = self.D_set["Ksv1"]
            f = self.D_set["f"]
            partial = self.D_set.get("sv_partial", False)

            if not partial:
                Ksv2 = self.D_set["Ksv2"]

        elif fit_type == "Temperature":

            from fitting_models import T_from_tau_arrhenius

            k0 = self.D_set["k0"]
            k1 = self.D_set["k1"]
            deltaE = self.D_set["deltaE"]

        elif fit_type == "pH":

            required = ["Top", "Bottom", "pKa", "slope"]
            for k in required:
                if k not in self.D_set:
                    raise ValueError(
                        "Please run pH fitting first."
                    )

            Top = self.D_set["Top"]
            Bottom = self.D_set["Bottom"]
            pKa = self.D_set["pKa"]
            slope = self.D_set["slope"]

        else:
            raise ValueError(
                f"Video generation not implemented for '{fit_type}'."
            )

        with h5py.File(h5_file, "r") as u:
            groups = sorted(
                [g for g in u.keys() if g.startswith("gates_")]
            )

        groups = self.get_selected_groups(groups)

        if groups is None or len(groups) == 0:
            return

        bg = functions.hdf5_read(h5_file, "background")
        bg1 = bg["background1"]
        bg2 = bg["background2"]

        try:
            thresh = float(self.thresh_box.text())
        except ValueError:
            thresh = 0
            self.thresh_box.setText("0")

        try:
            value_vmin = float(self.amin_box.text())
            value_vmax = float(self.amax_box.text())
        except ValueError:
            raise ValueError(
                "Please enter valid min/max values."
            )

        fig = plt.figure(figsize=(6, 6), constrained_layout=True)

        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        ax_img = fig.add_subplot(111)

        ax_img.set_xlabel("x (px)")
        ax_img.set_ylabel("y (px)")

        # First frame
        first = functions.hdf5_read(h5_file, groups[0])

        G1 = first["G1"] - bg1
        G2 = first["G2"] - bg2

        ratio = G2 / (G1 + 1e-9)
        ratio = np.clip(ratio, 1e-9, 0.999)

        tau = -t1 / np.log(ratio)

        if fit_type == "Oxygen":

            if partial:
                value_map = fitting_models.pO2_from_tau_partial(
                    tau,
                    tau0,
                    Ksv1,
                    f
                )
            else:
                value_map = fitting_models.pO2_from_tau(
                    tau,
                    tau0,
                    Ksv1,
                    Ksv2,
                    f
                )


            unit = self.D_set.get("oxygen_unit", "%")
            if unit == "%":
                cbar_label = f"Oxygen ({unit})"
            else: 
                cbar_label = f"Oxygen ({unit})"

        elif fit_type == "Temperature":

            T_K = T_from_tau_arrhenius(
                tau,
                k0,
                k1,
                deltaE
            )

            display_unit = self.D_set.get(
                "temperature_unit",
                "°C"
            )

            if display_unit == "°C":
                value_map = T_K - 273.15
                unit = "°C"
            else:
                value_map = T_K
                unit = "K"

            cbar_label = f"Temperature ({unit})"
        
        elif fit_type == "pH":

            ratio = G2 / (G1 + 1e-9)

            value_map = fitting_models.pH_from_ratio_boltz(
                ratio,
                Top,
                Bottom,
                pKa,
                slope
            )

            unit = ""
            cbar_label = "pH"

        mask = (G1 < thresh) | (G2 < thresh)

        value_map = value_map.astype(np.float64)
        value_map[mask] = np.nan
        value_map[~np.isfinite(value_map)] = np.nan

        display_map = apply_crop(value_map)

        value_roi, _ = apply_roi(value_map)

        mean_value = np.nanmean(value_roi)
        std_value = np.nanstd(value_roi)

        extent = None

        if "mm_per_pixel" in self.D_set:

            mm = self.D_set["mm_per_pixel"]

            h, w = value_map.shape

            extent = [0, w * mm, h * mm, 0]
        
        if extent is not None:
            ax_img.set_xlabel("x (mm)")
            ax_img.set_ylabel("y (mm)")

        im = ax_img.imshow(
            display_map,
            cmap="inferno",
            vmin=value_vmin,
            vmax=value_vmax,
            extent=extent,
            origin="upper"
        )

        fit_type = str(self.D_set["fit_type"]).strip()

        if fit_type.lower() == "ph":
            ax_img.set_title("pH")
        elif unit == "%":
            ax_img.set_title(f"{fit_type} [{unit}]")
        elif unit:
            ax_img.set_title(f"{fit_type} [{unit}]")
        else:
            ax_img.set_title(fit_type)

        cbar = fig.colorbar(im, ax=ax_img)
        cbar.set_label(cbar_label)

        txt = fig.text(
            0.5,
            0.03,
            f"Mean {fit_type} = {mean_value:.2f} {unit}, "
            f"STD = {std_value:.2f} {unit}",
            ha="center",
            fontsize=14
        )

        fig.canvas.draw()

        frame = np.array(
            fig.canvas.buffer_rgba(),
            dtype=np.uint8
        )

        h, w = frame.shape[:2]

        out = cv2.VideoWriter(
            video_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        for group_name in groups:

            data = functions.hdf5_read(
                h5_file,
                group_name
            )

            G1 = data["G1"] - bg1
            G2 = data["G2"] - bg2

            ratio = functions.hdf5_read(
                h5_file,
                group_name.replace("gates_", "")
            )["Ratio"]

            ratio = ratio.astype(np.float64)
            ratio = np.clip(ratio, 1e-9, 0.999)

            tau = -t1 / np.log(ratio)
            tau[~np.isfinite(tau)] = np.nan

            if fit_type == "Oxygen":

                if partial:
                    value_map = fitting_models.pO2_from_tau_partial(
                        tau,
                        tau0,
                        Ksv1,
                        f
                    )
                else:
                    value_map = fitting_models.pO2_from_tau(
                        tau,
                        tau0,
                        Ksv1,
                        Ksv2,
                        f
                    )

            elif fit_type == "Temperature":

                T_K = T_from_tau_arrhenius(
                    tau,
                    k0,
                    k1,
                    deltaE
                )

                if unit == "°C":
                    value_map = T_K - 273.15
                else:
                    value_map = T_K
            
            elif fit_type == "pH":

                ratio = G2 / (G1 + 1e-9)

                value_map = fitting_models.pH_from_ratio_boltz(
                    ratio,
                    Top,
                    Bottom,
                    pKa,
                    slope
                )

            mask = (G1 < thresh) | (G2 < thresh)

            value_map = value_map.astype(np.float64)
            value_map[mask] = np.nan
            value_map[~np.isfinite(value_map)] = np.nan

            value_roi, _ = apply_roi(value_map)

            mean_value = np.nanmean(value_roi)
            std_value = np.nanstd(value_roi)

            display_map = apply_crop(value_map)
            im.set_data(display_map)

            if fit_type == "pH":
                txt.set_text(
                f"Mean {fit_type} = {mean_value:.2f}, "
                f"STD = {std_value:.2f}"
            )
            else:
                txt.set_text(
                f"Mean {fit_type} = {mean_value:.2f} {unit}, "
                f"STD = {std_value:.2f} {unit}"
            )

            fig.canvas.draw()

            frame = np.array(
                fig.canvas.buffer_rgba(),
                dtype=np.uint8
            )

            out.write(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGBA2BGR
                )
            )

        out.release()
        plt.close(fig)

    def get_analysis_range(self, data, physical_max=None):
        try:
            vmin = float(self.amin_box.text())
            vmax = float(self.amax_box.text())
            if vmax > vmin:
                return vmin, vmax
        except ValueError:
            pass

        finite = np.isfinite(data)
        if not np.any(finite):
            return None, None

        vmin = 0
        vmax = np.nanpercentile(data[finite], 90)

        if physical_max is not None:
            vmax = min(vmax, physical_max)

        return vmin, vmax
    
    def set_default_analysis_range(self):
        fit_type = self.D_set.get("fit_type", "")

        defaults = {
            "Oxygen":      (0.0, 100.0),
            "Temperature": (0.0, 60.0),
            "pH":          (6.0, 13.0),
        }

        if fit_type not in defaults:
            return

        vmin, vmax = defaults[fit_type]

        self.amin_box.setText(f"{vmin:.1f}")
        self.amax_box.setText(f"{vmax:.1f}")

        if fit_type == "Oxygen":
            self.analysis_range_group.setTitle("pO2 Range")
        elif fit_type == "Temperature":
            self.analysis_range_group.setTitle("Temperature Range (°C)")
        elif fit_type == "pH":
            self.analysis_range_group.setTitle("pH Range")
        
