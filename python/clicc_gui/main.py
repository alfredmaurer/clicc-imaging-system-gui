import sys
import os
from pathlib import Path
import serial
import time
import numpy as np

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QStackedWidget
from ximea import xiapi

import functions
from globals_config import D_set
from pages.acquisition_page import AcquisitionGUI
from pages.calibration_page import CalibrationGUI
from pages.fit_page import FitGUI
from pages.analysis_page import AnalysisGUI
from pages.live_view_page import LiveViewGUI

BASE_DIR = Path(__file__).resolve().parent
save_dir = BASE_DIR / "experiment"
save_dir.mkdir(parents=True, exist_ok=True)

D_set["dir_path"] = str(save_dir)

class MainWindow(QWidget):
    def __init__(self, run_function, D_set): 
        super().__init__()

        self.stack = QStackedWidget()

        self.acq_page = AcquisitionGUI(run_function, D_set)
        self.stack.addWidget(self.acq_page)

        self.calibration_page = CalibrationGUI(D_set)
        self.stack.addWidget(self.calibration_page)

        self.fit_page = FitGUI(D_set)
        self.stack.addWidget(self.fit_page)

        self.analysis_page = AnalysisGUI(D_set)
        self.stack.addWidget(self.analysis_page)

        self.live_page = LiveViewGUI(D_set)
        self.stack.addWidget(self.live_page)

        self.btn_acq = QPushButton("Acquisition")
        self.btn_acq.clicked.connect(self.show_acquisition)

        self.btn_calibration = QPushButton("Calibration")
        self.btn_calibration.clicked.connect(self.show_calibration)

        self.btn_fit = QPushButton("Fitting")
        self.btn_fit.clicked.connect(self.show_fitting)

        self.btn_analysis = QPushButton("Analysis")
        self.btn_analysis.clicked.connect(self.show_analysis)

        self.btn_live = QPushButton("Live View")
        self.btn_live.clicked.connect(self.show_live)

        self.active_button_style = """
            QPushButton {
                color: #a184f5;
                font-weight: bold;
                background: transparent;
                border: none;
                border-bottom: 2px solid #a184f5;
                padding-bottom: 4px;
            }
        """

        self.inactive_button_style = """
            QPushButton {
                color: #7a7979;
                font-weight: normal;
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                padding-bottom: 4px;
            }
        """

        menu_bar = QHBoxLayout()
        menu_bar.addWidget(self.btn_acq)
        menu_bar.addWidget(self.btn_calibration)
        menu_bar.addWidget(self.btn_fit)
        menu_bar.addWidget(self.btn_analysis)
        menu_bar.addWidget(self.btn_live)
        menu_bar.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(menu_bar)
        layout.addWidget(self.stack)

        self.setLayout(layout)
        self.setWindowTitle("CLICC System")

        self.btn_acq.setStyleSheet(self.active_button_style)
        self.btn_calibration.setStyleSheet(self.inactive_button_style)
        self.btn_fit.setStyleSheet(self.inactive_button_style)
        self.btn_analysis.setStyleSheet(self.inactive_button_style)
        self.btn_live.setStyleSheet(self.inactive_button_style)

    def show_acquisition(self):
        self.stack.setCurrentWidget(self.acq_page)
        self.resize(1000, 500)

        self.btn_acq.setStyleSheet(self.active_button_style)
        self.btn_calibration.setStyleSheet(self.inactive_button_style)
        self.btn_fit.setStyleSheet(self.inactive_button_style)
        self.btn_analysis.setStyleSheet(self.inactive_button_style)
        self.btn_live.setStyleSheet(self.inactive_button_style)

    def show_calibration(self):
        self.calibration_page.refresh_path()
        self.calibration_page.refresh_button_text()
        self.stack.setCurrentWidget(self.calibration_page)

        self.btn_acq.setStyleSheet(self.inactive_button_style)
        self.btn_calibration.setStyleSheet(self.active_button_style)
        self.btn_fit.setStyleSheet(self.inactive_button_style)
        self.btn_analysis.setStyleSheet(self.inactive_button_style)
        self.btn_live.setStyleSheet(self.inactive_button_style)

    def show_fitting(self):
        self.fit_page.update_parameter_visibility()
        self.fit_page.calibration_Q = self.calibration_page.calibration_Q
        self.fit_page.calibration_tau = self.calibration_page.calibration_tau
        self.stack.setCurrentWidget(self.fit_page)

        self.btn_fit.setStyleSheet(self.active_button_style)
        self.btn_calibration.setStyleSheet(self.inactive_button_style)
        self.btn_acq.setStyleSheet(self.inactive_button_style)
        self.btn_analysis.setStyleSheet(self.inactive_button_style)
        self.btn_live.setStyleSheet(self.inactive_button_style)

    def show_analysis(self):
        self.stack.setCurrentWidget(self.analysis_page)

        self.btn_acq.setStyleSheet(self.inactive_button_style)
        self.btn_analysis.setStyleSheet(self.active_button_style)
        self.btn_calibration.setStyleSheet(self.inactive_button_style)
        self.btn_fit.setStyleSheet(self.inactive_button_style)
        self.btn_live.setStyleSheet(self.inactive_button_style)

        self.analysis_page.set_roi_from_dset()

    def show_live(self):
        self.stack.setCurrentWidget(self.live_page)

        self.btn_live.setStyleSheet(self.active_button_style)
        self.btn_acq.setStyleSheet(self.inactive_button_style)
        self.btn_calibration.setStyleSheet(self.inactive_button_style)
        self.btn_fit.setStyleSheet(self.inactive_button_style)
        self.btn_analysis.setStyleSheet(self.inactive_button_style)

def start_gui(run_function, D_set):
    app = QApplication(sys.argv)
    window = MainWindow(run_function, D_set)
    window.show()
    sys.exit(app.exec())

def run_acquisition(gui_ref):

    stm = serial.Serial(port='COM3', baudrate=9600)
    cam = functions.set_xiapi_parameters(50) 

    img = xiapi.Image()
    cam.set_exposure_burst_count(D_set['number_exposure'])

    gui_ref.cam = cam

    cam.start_acquisition()

    L_image_G1=[]
    L_image_G2=[]
    image_time_expo_camera_SPOT=[]

    image_background_1, image_background_2= functions.background(D_set, cam, stm, img)
    for _ in range(5) : 
        mode = D_set["mode"]
        image_G1, image_G2, t_G1, t_G2 = functions.take_one_image_ratio(D_set, cam, stm, img, mode) 
    
    period = D_set['cycle_delay']

    for i in range(D_set['nbr_shots_total']) :
        t_cycle_start = time.time()
        image_G1, image_G2, t_G1, t_G2 = functions.take_one_image_ratio(D_set, cam, stm, img, mode) 

        L_image_G1.append(image_G1)
        L_image_G2.append(image_G2)
        image_time_expo_camera_SPOT.append([t_G1,t_G2])
        print(i)

        if period > 0:
            elapsed = time.time() - t_cycle_start
            wait_time = max(0, period - elapsed)
            if wait_time > 0:
                time.sleep(wait_time)
        
        t_cycle_end = time.time()
        real_cycle_time = t_cycle_end - t_cycle_start
        print(f"Cycle {i}: Real duration = {real_cycle_time:.3f} seconds")

    cam.stop_acquisition()
    image_time_expo_camera_SPOT = np.array(image_time_expo_camera_SPOT)
    fps=1/(np.mean(np.diff(image_time_expo_camera_SPOT))*2)

    save_path = os.path.join(D_set['dir_path'], D_set['name'])

    D_set_list = [[k, [v]] for k, v in D_set.items()]

    functions.hdf5_append(
        save_path, 'parameters',
        D_set_list + [['times', image_time_expo_camera_SPOT]] +
        [['fps_measured', [fps]]] +
        [['nbImages', [D_set['nbr_shots_total']]]]
    )

    functions.hdf5_append(
        save_path, 'background',
        [['background1', image_background_1], ['background2', image_background_2]]
    )

    for i in range(D_set['nbr_shots_total']):
        functions.hdf5_append(
            save_path,
            'gates_' + format(i, '06.0f'),
            [['G1', L_image_G1[i]], ['G2', L_image_G2[i]]]
        )

        Ratio = (L_image_G2[i] - image_background_2) / (L_image_G1[i] - image_background_1)
        functions.hdf5_append(
            save_path,
            format(i, '06.0f'),
            [['Ratio', Ratio]]
        )

    stm.close()
    final_temp = None
    try:
        final_temp = cam.get_param(xiapi.XI_PRM_TEMP)
    except:
        pass

    cam.close_device()

    gui_ref.final_temperature = final_temp 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(run_acquisition, D_set)
    win.show()
    sys.exit(app.exec())
