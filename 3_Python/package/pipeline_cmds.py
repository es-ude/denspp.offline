from os import mkdir
from os.path import exists, join
from shutil import copy
from datetime import datetime
from scipy.io import savemat


class PipelineCMD:
    path2save: str
    _path2pipe: str

    def __init__(self):
        pass

    def generate_folder(self, path2runs: str, addon: str) -> None:
        """Generating the default folder for saving figures and data"""
        str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'{str_datum}_pipeline{addon}'

        if not exists(path2runs):
            mkdir(path2runs)

        path2save = join(path2runs, folder_name)
        if not exists(path2save):
            mkdir(path2save)

        copy(src=self._path2pipe, dst=path2save)
        self.path2save = path2save

    def save_results(self, name: str, data: dict) -> None:
        """Saving the data with a dictionary"""
        path2data = join(self.path2save, name)
        savemat(path2data, data)
        print(f"... data saved in: {path2data}")


class PipelineSignal:
    def __init__(self) -> None:
        self.u_in = None  # Input voltage
        self.u_pre = None  # Output of pre-amp
        self.u_spk = None  # Output of analogue filtering - spike acitivity
        self.u_lfp = None  # Output of analogue filtering - lfp
        self.x_adc = None  # ADC output
        self.fs_ana = 0.0  # "Sampling rate"

        self.x_adc = None  # ADC output
        self.x_spk = None  # Output of digital filtering - spike
        self.x_lfp = None  # Output of digital filtering - lfp
        self.x_sda = None  # Output of Spike Detection Algorithm (SDA)
        self.x_thr = None  # Threshold value for SDA
        self.x_pos = None  # Position for generating frames
        self.frames_orig = None  # Original frames after event-detection (larger)
        self.frames_align = None  # Aligned frames to specific method
        self.features = None  # Calculated features of frames
        self.cluster_id = None  # Clustered events
        self.spike_ticks = None  # Spike Ticks
        self.nsp_post = dict()  # Adding some parameters after calculating some neural signal processing methods
        self.fs_adc = 0.0  # Sampling rate of the ADC incl. oversampling
        self.fs_dig = 0.0  # Processing rate of the digital part

        self.spike_ticks = None  # Spike Ticks
        self.nsp_post = dict()  # Adding some parameters after calculating some neural signal processing methods


