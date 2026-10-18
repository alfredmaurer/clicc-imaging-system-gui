import os
from io import BytesIO

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QHBoxLayout,
    QGroupBox, QFormLayout, QCheckBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from matplotlib.figure import Figure
 
from shared_widgets import CollapsibleGroup
import fitting_models

class FitGUI(QWidget):
    def __init__(self, D_set):
        super().__init__()

        self.D_set = D_set
        self.has_recalibration = False

        self.calibration_Q = np.array([])
        self.calibration_tau = np.array([])

        filename = D_set.get("name", "")
        if not filename.lower().endswith(".h5"):
            filename += ".h5"

        self.current_file = os.path.join(D_set["dir_path"], filename)
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        left_group = QGroupBox("Fit Controls")
        left_layout = QVBoxLayout(left_group)

        param_group = QGroupBox("Stern–Volmer parameters (Oxygen)")
        param_layout = QVBoxLayout()

        form = QFormLayout()

        self.chk_ksv2_zero = QCheckBox("Ksv₂ = 0 (partial model)")
        self.chk_ksv2_zero.setToolTip(
            "Use partial Stern–Volmer model:\n"
            "τ/τ₀ = f/(1+Ksv₁·pO₂) + (1−f)"
        )
        self.chk_ksv2_zero.setChecked(True)
        self.chk_ksv2_zero.stateChanged.connect(self.on_ksv2_zero_changed)

        param_layout.addWidget(self.chk_ksv2_zero)

        self.ksv1_box = QLineEdit()
        self.ksv2_box = QLineEdit()
        self.f_box = QLineEdit()
        self.tau0_box = QLineEdit()

        form.addRow("τ₀ (ms):", self.tau0_box)
        form.addRow("Ksv1:", self.ksv1_box)
        form.addRow("Ksv2:", self.ksv2_box)
        form.addRow("f:", self.f_box)

        self.tau0_box.setReadOnly(False)

        param_layout.addLayout(form)

        self.sv_group = CollapsibleGroup(
            "Stern–Volmer parameters (Oxygen)",
            param_layout,
            expanded=True
        )

        left_layout.addWidget(self.sv_group)

        param_group_arrh = QGroupBox("Arrhenius parameters (Temperature)")
        param_layout_arrh = QFormLayout()

        self.k0_box = QLineEdit()
        self.k1_box = QLineEdit()
        self.delta_e_box = QLineEdit()

        param_layout_arrh.addRow("k₀:", self.k0_box)
        param_layout_arrh.addRow("k₁:", self.k1_box)

        label_deltaE = QLabel("ΔE<sub>S1–T1</sub> (eV)")
        label_deltaE.setTextFormat(Qt.TextFormat.RichText)
        param_layout_arrh.addRow(label_deltaE, self.delta_e_box)


        self.arrh_group = CollapsibleGroup(
            "Arrhenius parameters (Temperature)",
            param_layout_arrh,
            expanded=False
        )

        left_layout.addWidget(self.arrh_group)

        param_group_sigm = QGroupBox("Sigmoidal parameters (pH)")
        param_layout_sigm = QFormLayout()

        self.Top_box = QLineEdit()
        self.Bottom_box = QLineEdit()
        self.pKa_box = QLineEdit()
        self.slope_box = QLineEdit()

        param_layout_sigm.addRow("Top:", self.Top_box)
        param_layout_sigm.addRow("Bottom:", self.Bottom_box)
        param_layout_sigm.addRow("pKa:", self.pKa_box)
        param_layout_sigm.addRow("slope:", self.slope_box)

        self.sigm_group = CollapsibleGroup(
            "Sigmoidal parameters (pH)",
            param_layout_sigm,
            expanded=False
        )

        left_layout.addWidget(self.sigm_group)

        self.btn_fit_start = QPushButton("Fit")
        self.btn_fit_start.clicked.connect(self.run_fit)

        left_layout.addWidget(self.btn_fit_start)

        self.btn_use_manual = QPushButton("Use Manual Parameters")
        self.btn_use_manual.clicked.connect(self.use_manual_parameters)
        left_layout.addWidget(self.btn_use_manual)

        self.recal_group = QGroupBox("2-Point Recalibration")
        recal_layout = QFormLayout()

        # inputs
        self.ref_tau1 = QLineEdit()
        self.ref_Q1 = QLineEdit()
        self.ref_tau2 = QLineEdit()
        self.ref_Q2 = QLineEdit()
        self.ref_tau3 = QLineEdit()
        self.ref_Q3   = QLineEdit()

        # row 1
        self.lbl_tau1 = QLabel("τ₁ (ms):")
        self.lbl_Q1   = QLabel("O₂₁:")

        row1 = QHBoxLayout()
        row1.addWidget(self.lbl_tau1)
        row1.addWidget(self.ref_tau1)
        row1.addWidget(self.lbl_Q1)
        row1.addWidget(self.ref_Q1)

        # row 2
        self.lbl_tau2 = QLabel("τ₂ (ms):")
        self.lbl_Q2   = QLabel("O₂₂:")

        row2 = QHBoxLayout()
        row2.addWidget(self.lbl_tau2)
        row2.addWidget(self.ref_tau2)
        row2.addWidget(self.lbl_Q2)
        row2.addWidget(self.ref_Q2)

        # row 3
        self.lbl_tau3 = QLabel("R₃:")
        self.lbl_Q3   = QLabel("pH₃:")

        row3 = QHBoxLayout()
        row3.addWidget(self.lbl_tau3)
        row3.addWidget(self.ref_tau3)
        row3.addWidget(self.lbl_Q3)
        row3.addWidget(self.ref_Q3)

        recal_layout.addRow(row1)
        recal_layout.addRow(row2)
        recal_layout.addRow(row3)

        # hide by default
        self.lbl_tau3.hide()
        self.ref_tau3.hide()
        self.lbl_Q3.hide()
        self.ref_Q3.hide()

        # button
        self.btn_recalibrate = QPushButton("2-Point Recalibration")
        self.btn_recalibrate.clicked.connect(self.run_recalibration)

        recal_layout.addRow(self.btn_recalibrate)

        self.recal_group.setLayout(recal_layout)

        left_layout.addWidget(self.recal_group)

        self.recal_group.setEnabled(False)

        left_layout.addStretch() 

        self.btn_reset_fit = QPushButton("Reset Fit")
        self.btn_reset_fit.clicked.connect(self.reset_fit)

        left_layout.addWidget(self.btn_reset_fit)

        right_group = QGroupBox("Figure Display")
        right_layout = QVBoxLayout()
        right_group.setLayout(right_layout)

        self.fig_label = QLabel("Figures will appear here")
        self.fig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fig_label.setStyleSheet("background: #EEEEEE; border: 1px solid #CCCCCC;")
        right_layout.addWidget(self.fig_label)

        main_layout.addWidget(left_group, stretch=1)
        main_layout.addWidget(right_group, stretch=2)

        self.tau0_box.editingFinished.connect(self.store_parameters_to_dset)
        self.ksv1_box.editingFinished.connect(self.store_parameters_to_dset)
        self.ksv2_box.editingFinished.connect(self.store_parameters_to_dset)
        self.f_box.editingFinished.connect(self.store_parameters_to_dset)

    def run_fit(self):

        self.manual_mode = False

        self.original_params = None
        self.old_params = None

        self.has_recalibration = False
        fit_type = self.D_set.get("fit_type", "Oxygen")

        main = self.window()
        if hasattr(main, "calibration_page"):
            self.calibration_Q = main.calibration_page.calibration_Q
            self.calibration_y = main.calibration_page.calibration_tau  
            self.calibration_ratio_std = getattr(main.calibration_page,"calibration_ratio_std",None)

        Q = self.calibration_Q
        y = self.calibration_y

        if len(Q) < 3:
            QMessageBox.warning(
                self,
                "Not enough data",
                "At least 3 calibration points are required."
            )
            return
    
        if fit_type == "Temperature":

            tau = y

            try:
                result = fitting_models.fit_arrhenius(Q, tau)
            except Exception as e:
                QMessageBox.critical(self, "Fit failed", str(e))
                return

            self.k0_box.setText(f"{result['k0']:.4e}")
            self.k1_box.setText(f"{result['k1']:.4e}")
            self.delta_e_box.setText(f"{result['deltaE']:.4f}")

            self.original_params = {
                "k0": result["k0"],
                "k1": result["k1"],
                "deltaE": result["deltaE"]
            }

            self.old_params = self.original_params.copy()

            self.D_set["k0"] = result["k0"]
            self.D_set["k1"] = result["k1"]
            self.D_set["deltaE"] = result["deltaE"]

            self.plot_temperature_fit(Q, tau, result)
            self.recal_group.setEnabled(True)
            return
        
        if fit_type == "pH":

            ratio = y

            if len(Q) < 4:
                QMessageBox.warning(
                    self, "Not enough data",
                    "At least 4 calibration points are required for pH fitting."
                )
                return

            try:
                result = fitting_models.fit_boltzmann_pH(Q, ratio)
            except Exception as e:
                QMessageBox.critical(self, "Fit failed", str(e))
                return

            self.Top_box.setText(f"{result['Top']:.4f}")
            self.Bottom_box.setText(f"{result['Bottom']:.4f}")
            self.pKa_box.setText(f"{result['pKa']:.3f}")
            self.slope_box.setText(f"{result['slope']:.3f}")

            self.original_params = {
                "Top": result["Top"],
                "Bottom": result["Bottom"],
                "pKa": result["pKa"],
                "slope": result["slope"]
            }

            for k in ("Top", "Bottom", "pKa", "slope"):
                self.D_set[k] = float(result[k])

            main = self.window()
            if hasattr(main, "analysis_page"):
                main.analysis_page.invalidate_pH_map()
            
            self.old_params = {
                "Top": result["Top"],
                "Bottom": result["Bottom"],
                "pKa": result["pKa"],
                "slope": result["slope"]
            }

            self.plot_pH_fit(Q, ratio, result)
            self.recal_group.setEnabled(True)
            return
        
        if fit_type == "Oxygen":

            def get_float(box):
                txt = box.text().strip()
                if txt == "":
                    return None
                return float(txt.replace(",", "."))

            fixed_f = get_float(self.f_box)

            tau = y

            try:
                result = fitting_models.fit_stern_volmer(
                    Q,
                    tau,
                    force_ksv2_zero=self.chk_ksv2_zero.isChecked(),
                    fixed_f=fixed_f
                )
            except Exception as e:
                QMessageBox.critical(self, "Fit failed", str(e))
                return

            self.tau0_box.setText(f"{result['tau0']:.6g}")
            self.ksv1_box.setText(f"{result['Ksv1']:.5f}")
            def format_ksv2(val):
                if abs(val) < 1e-6 * 1.0001:
                    return f"{val:.0e}"
                return f"{val:.5f}"

            self.ksv2_box.setText(format_ksv2(result["Ksv2"]))
            self.f_box.setText(f"{result['f']:.3f}")

            self.original_params = {
                "tau0": result["tau0"],
                "Ksv1": result["Ksv1"],
                "Ksv2": result["Ksv2"],
                "f": result["f"]
            }

            self.D_set["tau0"] = result["tau0"]
            self.D_set["tau0_initial"] = result["tau0"]

            self.old_params = {
                "tau0": result["tau0"],
                "Ksv1": result["Ksv1"],
                "Ksv2": result["Ksv2"],
                "f": result["f"],
                "partial": result["partial"]
            }

            self.store_parameters_to_dset()
            self.plot_oxygen_fit(Q, tau, result, recalibrated=False)
            self.recal_group.setEnabled(True)

    def use_manual_parameters(self):

        self.manual_mode = True

        fit_type = self.D_set.get("fit_type", "Oxygen")

        def get_float(box):
            txt = box.text().strip()
            if txt == "":
                return None
            return float(txt.replace(",", "."))

        if fit_type == "Oxygen":

            tau0 = get_float(self.tau0_box)
            Ksv1 = get_float(self.ksv1_box)
            Ksv2 = get_float(self.ksv2_box)
            f    = get_float(self.f_box)

            if tau0 is None or Ksv1 is None or f is None:
                QMessageBox.warning(self, "Missing parameters",
                                    "Please enter at least τ0, Ksv1 and f.")
                return

            partial = self.chk_ksv2_zero.isChecked() or (Ksv2 is not None and abs(Ksv2) < 1e-12)

            if partial:
                model = fitting_models.stern_volmer_partial_tau
                Ksv2 = 0.0
            else:
                if Ksv2 is None:
                    QMessageBox.warning(self, "Missing parameters", "Please enter Ksv2.")
                    return
                model = fitting_models.stern_volmer_two_site_tau

            result = {
                "tau0": tau0,
                "Ksv1": Ksv1,
                "Ksv2": Ksv2,
                "f": f,
                "model": model,
                "partial": partial
            }

            self.original_params = result.copy()
            self.old_params = result.copy()

            self.D_set["tau0"] = tau0
            self.D_set["tau0_initial"] = tau0
            self.D_set["Ksv1"] = Ksv1
            self.D_set["Ksv2"] = Ksv2
            self.D_set["f"] = f
            self.D_set["sv_partial"] = partial

            self.has_recalibration = False

            if hasattr(self, "calibration_Q") and len(self.calibration_Q) > 0:
                self.plot_oxygen_fit(self.calibration_Q, self.calibration_y, result)
            else:
                Q = np.linspace(0, 100, 200)
                tau_model = model(Q, tau0, Ksv1, f) if partial else model(Q, tau0, Ksv1, Ksv2, f)
                self.plot_oxygen_fit(Q, tau_model, result)

            self.recal_group.setEnabled(True)
            return
        
        if fit_type == "pH":

            Top    = get_float(self.Top_box)
            Bottom = get_float(self.Bottom_box)
            pKa    = get_float(self.pKa_box)
            slope  = get_float(self.slope_box)

            if None in (Top, Bottom, pKa, slope):
                QMessageBox.warning(self, "Missing parameters", "Fill all pH parameters.")
                return

            result = {
                "Top": Top,
                "Bottom": Bottom,
                "pKa": pKa,
                "slope": slope,
                "model": fitting_models.boltzmann_pH
            }

            self.original_params = result.copy()
            self.old_params = result.copy()

            self.D_set["Top"] = result["Top"]
            self.D_set["Bottom"] = result["Bottom"]
            self.D_set["pKa"] = result["pKa"]

            for k in result:
                if k != "model":
                    self.D_set[k] = result[k]

            self.has_recalibration = False

            if hasattr(self, "calibration_Q") and len(self.calibration_Q) > 0:
                self.plot_pH_fit(self.calibration_Q, self.calibration_y, result)
            else:
                pKa = result["pKa"]
                pH = np.linspace(pKa - 5, pKa + 5, 300)
                R = result["model"](pH, Top, Bottom, pKa, slope)
                self.plot_pH_fit(
                    np.array([]),  
                    np.array([]),
                    result
                    )

            self.recal_group.setEnabled(True)
            return

        if fit_type == "Temperature":

            k0 = get_float(self.k0_box)
            k1 = get_float(self.k1_box)
            deltaE = get_float(self.delta_e_box)

            if None in (k0, k1, deltaE):
                QMessageBox.warning(self, "Missing parameters", "Fill all temperature parameters.")
                return

            result = {
                "k0": k0,
                "k1": k1,
                "deltaE": deltaE,
                "model": fitting_models.arrhenius_tau
            }

            self.original_params = result.copy()
            self.old_params = result.copy()

            self.D_set["k0"] = k0
            self.D_set["k1"] = k1
            self.D_set["deltaE"] = deltaE

            self.has_recalibration = False

            if hasattr(self, "calibration_Q") and len(self.calibration_Q) > 0:
                self.plot_temperature_fit(self.calibration_Q, self.calibration_y, result)
            else:
                T = np.linspace(273, 350, 200)
                tau = result["model"](T, k0, k1, deltaE)
                self.plot_temperature_fit(T, tau, result)

            self.recal_group.setEnabled(True)
    
    def update_parameter_visibility(self):

        fit_type = self.D_set.get("fit_type", "Oxygen")

        for group in (self.sv_group, self.arrh_group, self.sigm_group):
            group.set_expanded(False)
            group.content.setEnabled(False)

        if fit_type == "Oxygen":
            self.sv_group.set_expanded(True)
            self.sv_group.content.setEnabled(True)

        elif fit_type == "Temperature":
            self.arrh_group.set_expanded(True)
            self.arrh_group.content.setEnabled(True)

        elif fit_type == "pH":
            self.sigm_group.set_expanded(True)
            self.sigm_group.content.setEnabled(True)
        
        self.update_recalibration_labels()
        self.clear_recalibration_inputs()
    
    def plot_oxygen_fit(self, Q, tau, result, recalibrated=False):
        tau0 = result["tau0"]
        tau0_data = self.D_set.get("tau0_initial", result["tau0"])
        y = tau0_data / tau

        Q_fit = np.linspace(Q.min(), Q.max(), 300)
        if result["partial"]:
            tau_model = result["model"](
                Q_fit,
                result["tau0"],
                result["Ksv1"],
                result["f"]
            )
        else:
            tau_model = result["model"](
                Q_fit,
                result["tau0"],
                result["Ksv1"],
                result["Ksv2"],
                result["f"]
            )

        # convert to τ0 / τ for plotting
        y_fit_new = result["tau0"] / tau_model

        y_fit_old = None

        Q_fit_old = np.linspace(Q.min(), Q.max(), 300)

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        if self.has_recalibration and hasattr(self, "old_params"):

            old = self.old_params

            if old["partial"]:
                tau_old = fitting_models.stern_volmer_partial_tau(
                    Q_fit,
                    old["tau0"],
                    old["Ksv1"],
                    old["f"]
                )
            else:
                tau_old = fitting_models.stern_volmer_two_site_tau(
                    Q_fit,
                    old["tau0"],
                    old["Ksv1"],
                    old["Ksv2"],
                    old["f"]
                )

            y_fit_old = old["tau0"] / tau_old

            ax.plot(Q_fit, y_fit_old, "--", color="gray", label="Original fit")


        show_data = (
            not getattr(self, "manual_mode", False)
            and hasattr(self, "calibration_Q")
            and len(self.calibration_Q) > 0
        )

        if show_data:
            std = getattr(self, "calibration_ratio_std", None)

            if std is not None and len(std) == len(Q):
                yerr = tau0 * std / (tau ** 2)

                ax.errorbar(
                    Q, y,
                    yerr=yerr,
                    fmt="o",
                    color="tab:blue",
                    ecolor="black",
                    capsize=4,
                    markersize=6,
                    label="Calibration data"
                )
            else:
                ax.scatter(Q, y, label="Calibration data")

        label = "Recalibrated fit" if self.has_recalibration else "Stern–Volmer fit"
        ax.plot(Q_fit, y_fit_new, color="red", label=label)

        unit = self.D_set.get("oxygen_unit", "%")

        if unit == "%":
            xlabel = "Oxygen (%)"
        elif unit == "kPa":
            xlabel = "Oxygen (kPa)"
        elif unit == "mmol/L":
            xlabel = "Oxygen (mmol/L)"
        else:
            xlabel = "Oxygen"

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(r"$\tau_0 / \tau$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

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
    
    def plot_temperature_fit(self, T, tau, result):

        T_fit = np.linspace(T.min(), T.max(), 300)

        tau_fit_new = result["model"](
            T_fit,
            result["k0"],
            result["k1"],
            result["deltaE"]
        )

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        if self.has_recalibration and hasattr(self, "old_params") and self.old_params:

            old = self.old_params

            tau_fit_old = result["model"](
                T_fit,
                old["k0"],
                old["k1"],
                old["deltaE"]
            )

            ax.plot(
                T_fit,
                tau_fit_old * 1e3,
                "--",
                color="gray",
                label="Original fit"
            )

        show_data = (
            not getattr(self, "manual_mode", False)
            and hasattr(self, "calibration_Q")
            and len(self.calibration_Q) > 0
        )

        if show_data:
            std = getattr(self, "calibration_ratio_std", None)

            if std is not None and len(std) == len(T):
                ax.errorbar(
                    T, tau * 1e3,
                    yerr=std * 1e3,
                    fmt="o",
                    color="tab:blue",
                    ecolor="black",
                    capsize=4,
                    markersize=6,
                    label="Calibration data"
                )
            else:
                ax.scatter(T, tau * 1e3, label="Calibration data")

        label = "Recalibrated fit" if self.has_recalibration else "Arrhenius fit"

        ax.plot(
            T_fit,
            tau_fit_new * 1e3,
            color="red",
            label=label
        )

        ax.set_xlabel("Temperature (K)", fontsize=14)
        ax.set_ylabel("Lifetime τ (µs)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

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

    def plot_pH_fit(self, pH, ratio, result, recalibrated=False):

        if len(pH) > 0:
            pH_min, pH_max = np.min(pH), np.max(pH)
        else:
            pKa = result["pKa"]
            pH_min, pH_max = pKa - 5, pKa + 5

        pH_fit = np.linspace(pH_min, pH_max, 400)

        R_fit_new = result["model"](
            pH_fit,
            result["Top"],
            result["Bottom"],
            result["pKa"],
            result["slope"]
        )

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        if self.has_recalibration and hasattr(self, "old_params") and self.old_params:

            old = self.old_params

            R_fit_old = result["model"](
                pH_fit,
                old["Top"],
                old["Bottom"],
                old["pKa"],
                old["slope"]
            )

            ax.plot(
                pH_fit,
                R_fit_old,
                "--",
                color="gray",
                label="Original fit"
            )

        show_data = (
            not getattr(self, "manual_mode", False)
            and hasattr(self, "calibration_Q")
            and len(self.calibration_Q) > 0
        )

        if show_data:
            ratio_std = getattr(self, "calibration_ratio_std", None)

            if ratio_std is not None and len(ratio_std) == len(pH):
                ax.errorbar(
                    pH, ratio,
                    yerr=ratio_std,
                    fmt='o',
                    color='tab:blue',
                    ecolor='black',
                    capsize=4,
                    markersize=6,
                    label="Calibration data"
                )
            else:
                ax.scatter(pH, ratio, label="Calibration data")

        label = "Recalibrated fit" if recalibrated else "Boltzmann fit"
        ax.plot(pH_fit, R_fit_new, color="red", label=label)

        ax.set_xlabel("pH", fontsize=14)
        ax.set_ylabel("Ratio (G2 / G1)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title("pH Calibration Fit")
        ax.grid(True, alpha=0.3)
        ax.legend()

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

    def on_ksv2_zero_changed(self, state):
            if state == Qt.CheckState.Checked:
                self.ksv2_box.setText("0.0")
                self.ksv2_box.setDisabled(True)
                self.D_set["sv_partial"] = True
            else:
                self.ksv2_box.setDisabled(False)
                self.D_set["sv_partial"] = False

    def store_parameters_to_dset(self):

        def parse_optional(box):
            txt = box.text().strip()
            if txt == "":
                return None
            txt = txt.replace(",", ".")
            return float(txt)

        try:
            tau0 = parse_optional(self.tau0_box)
            Ksv1 = parse_optional(self.ksv1_box)
            Ksv2 = parse_optional(self.ksv2_box)
            f    = parse_optional(self.f_box)

            if tau0 is not None:
                self.D_set["tau0"] = tau0
            if Ksv1 is not None:
                self.D_set["Ksv1"] = Ksv1
            if Ksv2 is not None:
                self.D_set["Ksv2"] = Ksv2
            if f is not None:
                self.D_set["f"] = f

            if "Ksv2" in self.D_set:
                self.D_set["sv_partial"] = abs(self.D_set["Ksv2"]) < 1e-12

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid parameters",
                "Please enter valid numeric Stern–Volmer parameters.\n"
                "Fields can be filled one by one."
            )

    def run_recalibration(self):
        fit_type = self.D_set.get("fit_type", "Oxygen")

        self.has_recalibration = True

        if fit_type == "Oxygen":

            if self.D_set.get("Ksv1") is None or self.D_set.get("f") is None:
                QMessageBox.warning(
                    self,
                    "Missing parameters",
                    "Run initial fit first (Ksv1 and f required)."
                )
                return

            def get_float(box):
                txt = box.text().strip()
                if txt == "":
                    return None
                return float(txt.replace(",", "."))

            tau1 = get_float(self.ref_tau1)
            tau2 = get_float(self.ref_tau2)
            Q1   = get_float(self.ref_Q1)
            Q2   = get_float(self.ref_Q2)

            if None in (tau1, tau2, Q1, Q2):
                QMessageBox.warning(self, "Missing data", "Enter both reference points")
                return
            
            orig = getattr(self, "original_params", None)

            tau0_init = orig["tau0"] if orig else self.D_set.get("tau0")
            Ksv1_init = orig["Ksv1"] if orig else self.D_set.get("Ksv1")

            try:
                result = fitting_models.recalibrate_tau0_ksv1(
                    tau_ref=[tau1, tau2],
                    Q_ref=[Q1, Q2],
                    Ksv2=self.D_set.get("Ksv2"),
                    f=self.D_set.get("f"),
                    tau0_init=tau0_init,   
                    Ksv1_init=Ksv1_init  
                )
            except Exception as e:
                QMessageBox.critical(self, "Recalibration failed", str(e))
                return

            self.tau0_box.setText(f"{result['tau0']:.6g}")
            self.ksv1_box.setText(f"{result['Ksv1']:.6g}")

            self.D_set["tau0"] = result["tau0"]
            self.D_set["Ksv1"] = result["Ksv1"]

            result_plot = {
                "tau0": self.D_set["tau0"],
                "Ksv1": self.D_set["Ksv1"],
                "Ksv2": self.D_set["Ksv2"],  
                "f": self.D_set["f"],
                "model": (
                    fitting_models.stern_volmer_partial_tau
                    if self.D_set.get("sv_partial", False)
                    else fitting_models.stern_volmer_two_site_tau
                ),
                "partial": self.D_set.get("sv_partial", False)
            }

            if (
                not getattr(self, "manual_mode", False)
                and hasattr(self, "calibration_Q")
                and len(self.calibration_Q) > 0
            ):
                Q = self.calibration_Q
                tau = self.calibration_y
            else:
                Q = np.linspace(0, 100, 200)

                if result_plot["partial"]:
                    tau = fitting_models.stern_volmer_partial_tau(
                        Q,
                        result_plot["tau0"],
                        result_plot["Ksv1"],
                        result_plot["f"]
                    )
                else:
                    tau = fitting_models.stern_volmer_two_site_tau(
                        Q,
                        result_plot["tau0"],
                        result_plot["Ksv1"],
                        result_plot["Ksv2"],
                        result_plot["f"]
                    )

            self.plot_oxygen_fit(
                Q,
                tau,
                result_plot,
                recalibrated=True
            )

            QMessageBox.information(
                self,
                "Recalibration successful",
                f"τ₀ = {result['tau0']:.6g}\nKsv1 = {result['Ksv1']:.6g}"
            )

            return
    
        if fit_type == "pH":

            def get_float(box):
                txt = box.text().strip()
                if txt == "":
                    return None
                return float(txt.replace(",", "."))

            R1 = get_float(self.ref_tau1)
            R2 = get_float(self.ref_tau2)
            R3 = get_float(self.ref_tau3)

            pH1 = get_float(self.ref_Q1)
            pH2 = get_float(self.ref_Q2)
            pH3 = get_float(self.ref_Q3)

            if None in (R1, R2, pH1, pH2):
                QMessageBox.warning(
                    self,
                    "Missing data",
                    "Enter at least 2 recalibration points."
                )
                return

            R_vals = [R1, R2]
            pH_vals = [pH1, pH2]

            if R3 is not None and pH3 is not None:
                R_vals.append(R3)
                pH_vals.append(pH3)

            self.old_params = self.original_params.copy()

            try:
                result = fitting_models.recalibrate_pH(
                    pH_ref=pH_vals,
                    R_ref=R_vals,
                    Top_init=self.original_params["Top"],
                    Bottom_init=self.original_params["Bottom"],
                    pKa_init=self.original_params["pKa"],
                    slope=self.original_params["slope"],
                    weight_prior=0.1
                )

            except Exception as e:
                QMessageBox.critical(self, "Recalibration failed", str(e))
                return

            self.Top_box.setText(f"{result['Top']:.4f}")
            self.Bottom_box.setText(f"{result['Bottom']:.4f}")
            self.pKa_box.setText(f"{result['pKa']:.4f}")

            self.D_set["Top"] = result["Top"]
            self.D_set["Bottom"] = result["Bottom"]
            self.D_set["pKa"] = result["pKa"]

            result_plot = {
                "Top": result["Top"],
                "Bottom": result["Bottom"],
                "pKa": result["pKa"],
                "slope": self.original_params["slope"],
                "model": fitting_models.boltzmann_pH
            }

            if (
                not getattr(self, "manual_mode", False)
                and hasattr(self, "calibration_Q")
                and len(self.calibration_Q) > 0
            ):
                pH = self.calibration_Q
                ratio = self.calibration_y
            else:
                pH = np.linspace(result["pKa"] - 5, result["pKa"] + 5, 300)
                ratio = fitting_models.boltzmann_pH(
                    pH,
                    result["Top"],
                    result["Bottom"],
                    result["pKa"],
                    result_plot["slope"]
                )

            self.plot_pH_fit(
                pH,
                ratio,
                result_plot,
                recalibrated=True
            )

            QMessageBox.information(
                self,
                "Recalibration successful",
                f"Top = {result['Top']:.4f}\n"
                f"Bottom = {result['Bottom']:.4f}\n"
                f"pKa = {result['pKa']:.4f}"
            )

            return

        if fit_type == "Temperature":

            def get_float(box):
                txt = box.text().strip()
                if txt == "":
                    return None
                return float(txt.replace(",", "."))

            tau1 = get_float(self.ref_tau1)
            tau2 = get_float(self.ref_tau2)
            T1   = get_float(self.ref_Q1)
            T2   = get_float(self.ref_Q2)

            if None in (tau1, tau2, T1, T2):
                QMessageBox.warning(self, "Missing data", "Enter both reference points")
                return

            orig = getattr(self, "original_params", None)

            if not hasattr(self, "original_params") or self.original_params is None:
                QMessageBox.warning(self, "Missing parameters", "Run initial fit first.")
                return

            k0_init = self.original_params["k0"]
            k1_init = self.original_params["k1"]
            deltaE  = self.original_params["deltaE"]

            self.old_params = self.original_params.copy()

            try:
                result = fitting_models.recalibrate_arrhenius_k0_k1(
                    tau_ref=[tau1, tau2],
                    T_ref=[T1, T2],
                    deltaE=deltaE,
                    k0_init=k0_init,
                    k1_init=k1_init
                )
            except Exception as e:
                QMessageBox.critical(self, "Recalibration failed", str(e))
                return

            # update GUI
            self.k0_box.setText(f"{result['k0']:.4e}")
            self.k1_box.setText(f"{result['k1']:.4e}")

            # update dataset
            self.D_set["k0"] = result["k0"]
            self.D_set["k1"] = result["k1"]

            result_plot = {
                "k0": self.D_set["k0"],
                "k1": self.D_set["k1"],
                "deltaE": self.D_set["deltaE"],
                "model": fitting_models.arrhenius_tau
            }

            if (
                not getattr(self, "manual_mode", False)
                and hasattr(self, "calibration_Q")
                and len(self.calibration_Q) > 0
            ):
                T = self.calibration_Q
                tau = self.calibration_y
            else:
                T = np.linspace(273, 350, 200)
                tau = fitting_models.arrhenius_tau(
                    T,
                    result_plot["k0"],
                    result_plot["k1"],
                    result_plot["deltaE"]
                )

            self.plot_temperature_fit(
                T,
                tau,
                result_plot
            )

            QMessageBox.information(
                self,
                "Recalibration successful",
                f"k0 = {result['k0']:.4e}\n"
                f"k1 = {result['k1']:.4e}"
            )
            return

    def reset_fit(self):

        self.tau0_box.clear()
        self.ksv1_box.clear()
        self.ksv2_box.clear()
        self.f_box.clear()

        for key in ("tau0", "Ksv1", "Ksv2", "f"):
            if key in self.D_set:
                self.D_set.pop(key)

        self.has_recalibration = False

        self.chk_ksv2_zero.setChecked(True)
        self.ksv2_box.setDisabled(False)

        if hasattr(self, "old_params"):
            self.old_params = None

        self.fig_label.setText("Fit reset. Run new fit.")
    
    def update_recalibration_labels(self):

        fit_type = self.D_set.get("fit_type", "Oxygen")

        if fit_type == "Oxygen":
            self.recal_group.setTitle("2-Point Recalibration")
            self.btn_recalibrate.setText("2-Point Recalibration")

            self.lbl_tau1.setText("τ₁ (ms):")
            self.lbl_tau2.setText("τ₂ (ms):")
            self.lbl_Q1.setText("O₂₁:")
            self.lbl_Q2.setText("O₂₂:")

            self.lbl_tau3.hide()
            self.ref_tau3.hide()
            self.lbl_Q3.hide()
            self.ref_Q3.hide()

        elif fit_type == "pH":
            self.recal_group.setTitle("3-Point Recalibration")
            self.btn_recalibrate.setText("3-Point Recalibration")

            self.lbl_tau1.setText("R₁:")
            self.lbl_tau2.setText("R₂:")
            self.lbl_Q1.setText("pH₁:")
            self.lbl_Q2.setText("pH₂:")

            self.lbl_tau3.setText("R₃:")
            self.lbl_Q3.setText("pH₃:")

            self.lbl_tau3.show()
            self.ref_tau3.show()
            self.lbl_Q3.show()
            self.ref_Q3.show()

        elif fit_type == "Temperature":
            self.recal_group.setTitle("2-Point Recalibration")
            self.btn_recalibrate.setText("2-Point Recalibration")

            self.lbl_tau1.setText("τ₁ (ms):")
            self.lbl_tau2.setText("τ₂ (ms):")
            self.lbl_Q1.setText("T₁ (K):")
            self.lbl_Q2.setText("T₂ (K):")

            self.lbl_tau3.hide()
            self.ref_tau3.hide()
            self.lbl_Q3.hide()
            self.ref_Q3.hide()

    def clear_recalibration_inputs(self):
        for box in (
            self.ref_tau1, self.ref_tau2, self.ref_tau3,
            self.ref_Q1, self.ref_Q2, self.ref_Q3
        ):
            box.clear()

    
