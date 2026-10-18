import numpy as np
import serial
import matplotlib.cm as cm

from globals_config import D_set

from PyQt6.QtCore import QThread, pyqtSignal

from ximea import xiapi
import functions


class AcquisitionThread(QThread):
    error_signal = pyqtSignal(str)
    done_signal = pyqtSignal()

    def __init__(self, run_function):
        super().__init__()
        self.run_function = run_function

    def run(self):
        try:
            self.run_function()
            self.done_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

class LEDTestThread(QThread):
    finished = pyqtSignal(np.ndarray, dict)
    error = pyqtSignal(str)

    def __init__(self, D_set):
        super().__init__()
        self.D_set = D_set

    def run(self):
        try:
            stm = serial.Serial(port='COM3', baudrate=9600)
            cam = functions.set_xiapi_parameters(self.D_set['number_exposure'])
            img = xiapi.Image()

            cam.set_exposure_burst_count(self.D_set['number_exposure'])
            cam.start_acquisition()

            if "timeout" not in self.D_set:
                self.D_set["timeout"] = 5000 

            for k in ("image_acq_time1", "image_acq_time2"):
                if k not in self.D_set:
                    raise RuntimeError(f"{k} missing in D_set")

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

            frames = []

            print(self.D_set["mode"])

            if self.D_set["mode"] in ["RLD", "RLD overlapping"]:
                stm.write(bytes(str(1) + D_set['parameters_to_send'], 'utf-8'))
                for i in range(5):
                    stm.write(bytes(str(1), 'utf-8'))
                    cam.get_image(img, timeout =D_set['timeout'])
                    image_G1 = img.get_image_data_numpy()
                    print(image_G1.shape, image_G1.dtype, np.min(image_G1), np.max(image_G1), np.mean(image_G1))
                    frames.append(image_G1)
            
            else: 
                stm.write(bytes(str(5) + D_set['parameters_to_send'], 'utf-8'))
                for i in range(5):
                    stm.write(bytes(str(5), 'utf-8'))
                    cam.get_image(img, timeout =D_set['timeout'])
                    image_G1 = img.get_image_data_numpy()
                    print(image_G1.shape, image_G1.dtype, np.min(image_G1), np.max(image_G1), np.mean(image_G1))
                    frames.append(image_G1)
            
            img_norm = image_G1.astype(np.float32)
            img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-12)

            rgb = (cm.viridis(img_norm)[:, :, :3] * 255).astype(np.uint8)

            cam.stop_acquisition()
            cam.close_device()
            stm.close()

            mean_img = np.mean(frames, axis=0)

            mean_img_f = mean_img.astype(np.float32)

            mean_val = float(np.nanmean(mean_img_f))
            std_val  = float(np.nanstd(mean_img_f))
            cv = std_val / mean_val if mean_val > 0 else np.nan

            max_val = float(np.nanmax(mean_img_f))
            min_val = float(np.nanmin(mean_img_f))

            sat_thresh = 4094
            sat_ratio = float(np.mean(mean_img_f >= sat_thresh))

            metrics = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "cv": cv,
                "sat_thresh": sat_thresh,
                "saturation_pct": sat_ratio * 100
            }

            self.finished.emit(mean_img, metrics)

        except Exception as e:
            self.error.emit(str(e))

class LiveCameraThread(QThread):
    new_frame = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.running = True

    def run(self):
        img = xiapi.Image()

        try:
            while self.running:
                self.cam.get_image(img)
                frame = img.get_image_data_numpy()
                self.new_frame.emit(frame)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()


