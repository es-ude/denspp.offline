from os import mkdir, getcwd
from os.path import exists, join
from shutil import copy
from datetime import datetime
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, active_count
from tqdm import tqdm
import dataclasses


class PipelineCMD:
    path2save: str
    _path2pipe: str

    def __init__(self, path2start='3_Python') -> None:
        self._path2start = join(getcwd().split(path2start)[0], path2start)

    def get_pipeline_name(self) -> str:
        """Getting the name of the pipeline"""
        return self.__class__.__name__

    def generate_folder(self, path2runs: str, addon: str) -> None:
        """Generating the default folder for saving figures and data"""
        str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'{str_datum}_{self.get_pipeline_name().lower()}{addon}'

        path2start = join(self._path2start, path2runs)
        if not exists(path2start):
            mkdir(path2start)

        path2save = join(path2start, folder_name)
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


class ThreadProcessor(Thread):
    output_full = None
    output_save = None

    def __init__(self, rawdata: np.ndarray, fs_ana: float, pipeline):
        Thread.__init__(self)
        self.input = rawdata
        self.pipeline = pipeline(fs_ana)

    def run(self):
        self.pipeline.run(self.input)
        self.output_full = self.pipeline.signals
        self.output_save = self.pipeline.prepare_saving()


@dataclasses.dataclass
class ThreadSettings:
    use_multithreading: bool
    num_max_workers: int
    block_plots: bool
    fs_ana: float
    pipeline: None


class ProcessingData:
    def __init__(self, settings: ThreadSettings, data_in):
        # --- Preparing data
        self.data = data_in
        self._channel_id = data_in.electrode_id
        self.results_full = dict()
        self.results_save = dict()

        # --- Preparing threads handler
        self._settings = settings
        self.__threads_worker = list()
        self._max_num_workers = settings.num_max_workers
        self._max_num_channel = len(self._channel_id)
        self._num_iterations = int(np.ceil(self._max_num_channel / self._max_num_workers))

    def __perform_single_threads(self) -> None:
        """"""
        self.__threads_worker = list()
        print('... processing data via single threading')
        for idx, elec in enumerate(tqdm(self._channel_id, ncols=100, desc='Progress: ')):
            self.__threads_worker.append(ThreadProcessor(self.data.data_raw[idx], self._settings.fs_ana,
                                                         self._settings.pipeline))
            self.__threads_worker[idx].start()
            self.__threads_worker[idx].join()
            self.results_full.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_full})
            self.results_save.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_save})

    def __perform_multi_threads(self) -> None:
        """"""
        self.__threads_worker = list()
        process_threads = list()
        for ite in range(self._num_iterations):
            process_threads.append(self._channel_id[ite * self._max_num_workers : (ite + 1) * self._max_num_workers])

        print(f"... processing data with {self._max_num_workers} of {active_count()} threading workers")
        for thr in tqdm(process_threads, ncols=100, desc='Progress: '):
            self.__threads_worker = list()
            # --- Starting all threads
            for idx, elec in enumerate(thr):
                self.__threads_worker.append(ThreadProcessor(self.data.data_raw[idx], self._settings.fs_ana,
                                                             self._settings.pipeline))
                self.__threads_worker[idx].start()

            # --- Waiting all threads are ready
            for idx, elec in enumerate(thr):
                self.__threads_worker[idx].join()
                self.results_full.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_full})
                self.results_save.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_save})

    def do_save_results(self, do_matfile=False) -> None:
        """"""
        path2save = self.__threads_worker[-1].pipeline.path2save
        if not do_matfile:
            np.save(f'{path2save}/results_half.npy', self.results_save)
        else:
            savemat(f'{path2save}/results_half.mat', self.results_save, do_compression=True)

    def do_plot_results(self) -> None:
        """"""
        plt.close('all')
        for idx, elec_num in enumerate(self._channel_id):
            self.__threads_worker[-1].pipeline.do_plotting(self.results_full[f'Elec_{elec_num:03d}'], elec_num)

        plt.show(block=self._settings.block_plots)

    def do_processing(self) -> None:
        """"""
        if self._settings.use_multithreading and self._max_num_channel > 1:
            self.__perform_multi_threads()
        else:
            self.__perform_single_threads()
