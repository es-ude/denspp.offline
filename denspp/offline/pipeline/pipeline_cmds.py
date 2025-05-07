from os import mkdir
from os.path import exists, join
from shutil import copy
from datetime import datetime
from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, active_count
from tqdm import tqdm
from dataclasses import dataclass

from denspp.offline.structure_builder import get_path_project_start


class PipelineCMD:
    path2save: str
    _path2pipe: str

    def __init__(self) -> None:
        """Class for handling the pipeline processing"""
        self._logger = getLogger(__name__)
        self._path2start = get_path_project_start()

    def get_pipeline_name(self) -> str:
        """Getting the name of the pipeline"""
        return self.__class__.__name__

    def generate_folder(self, path2runs: str, addon: str) -> None:
        """Generating the default folder for saving figures and data
        Args:
            path2runs:      Main folder in which the figures and data is stored
            addon:          Name of new folder for saving results
        Returns:
            None
        """
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
        """Saving the data with a dictionary
        Args:
            name:   File name for saving results
            data:   Dictionary with data content
        Returns:
            None
        """
        path2data = join(self.path2save, name)
        np.save(path2data, data, allow_pickle=False)
        self._logger.info(f"... data saved in: {path2data}")


class ThreadProcessor(Thread):
    def __init__(self, rawdata: np.ndarray, fs_ana: float, pipeline) -> None:
        """Class for handling a thread of the signal processor
        Args:
            rawdata:    Numpy array of input rawdata
            fs_ana:     Sampling rate of data
            pipeline:   Pipeline/Thread Processor
        Returns:
            None
        """
        super().__init__()
        self.input = rawdata
        self.pipeline = pipeline(fs_ana)
        self.output_full = None
        self.output_save = None

    def run(self):
        """Do data processing"""
        self.pipeline.run(self.input)
        self.output_full = self.pipeline.signals
        self.output_save = self.pipeline.prepare_saving()


@dataclass(frozen=True)
class SettingsThread:
    """Class for handling the processor"""
    use_multithreading: bool
    num_max_workers: int
    block_plots: bool


RecommendedSettingsThread = SettingsThread(
    use_multithreading=False,
    num_max_workers=1,
    block_plots=True
)


class ProcessingData:
    def __init__(self, pipeline, settings: SettingsThread, data_in: np.ndarray, channel_id: np.ndarray, fs: float) -> None:
        """Thread processor for analyzing data
        Args:
            pipeline:       Used pipeline for signal processing
            settings:       Settings for handling the threads
            data_in:        Numpy array of input data for signal processing
            channel_id:     Corresponding ID of used electrode / channel
            fs:             Sampling rate of data
        Returns:
            None
        """
        self._logger = getLogger(__name__)
        # --- Preparing data
        self.data = data_in
        self.sampling_rate = fs
        self._channel_id = channel_id
        self.results_full = dict()
        self.results_save = dict()

        # --- Preparing threads handler
        self._pipeline = pipeline
        self._settings = settings
        self.__threads_worker = list()
        self._max_num_workers = settings.num_max_workers
        self._max_num_channel = len(self._channel_id)
        self._num_iterations = int(np.ceil(self._max_num_channel / self._max_num_workers))

    def __perform_single_threads(self) -> None:
        """Handler for processing all channels with one single thread"""
        self.__threads_worker = list()
        self._logger.info('... processing data via single threading')
        for idx, elec in enumerate(tqdm(self._channel_id, ncols=100, desc='Progress: ')):
            self.__threads_worker.append(ThreadProcessor(self.data[idx], self.sampling_rate, self._pipeline))
            self.__threads_worker[idx].start()
            self.__threads_worker[idx].join()
            self.results_full.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_full})
            self.results_save.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_save})

    def __perform_multi_threads(self) -> None:
        """Handler for processing all channels with several threads simultaneously"""
        self.__threads_worker = list()
        process_threads = list()
        for ite in range(self._num_iterations):
            process_threads.append(self._channel_id[ite * self._max_num_workers : (ite + 1) * self._max_num_workers])

        self._logger.info(f"... processing data with {self._max_num_workers} of {active_count()} threading workers")
        for thr in tqdm(process_threads, ncols=100, desc='Progress: '):
            self.__threads_worker = list()
            # --- Starting all threads
            for idx, elec in enumerate(thr):
                self.__threads_worker.append(ThreadProcessor(self.data[idx], self._settings.fs_ana, self.pipeline))
                self.__threads_worker[idx].start()

            # --- Waiting all threads are ready
            for idx, elec in enumerate(thr):
                self.__threads_worker[idx].join()
                self.results_full.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_full})
                self.results_save.update({f'Elec_{elec:03d}': self.__threads_worker[idx].output_save})

    def do_save_results(self) -> None:
        """Saving results in desired numpy format"""
        path2save = self.__threads_worker[-1].pipeline.path2save
        np.save(f'{path2save}/results_half.npy', self.results_save)

    def do_plot_results(self) -> None:
        """Plotting the results of all signal processors"""
        plt.close('all')
        for key in self.results_full.keys():
            self.__threads_worker[-1].pipeline.do_plotting(self.results_full[key], int(key[-3:]))

        plt.show(block=self._settings.block_plots)

    def do_processing(self) -> None:
        """Performing the data processing"""
        if self._settings.use_multithreading and self._max_num_channel > 1:
            self.__perform_multi_threads()
        else:
            self.__perform_single_threads()
