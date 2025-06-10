import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join
from shutil import copy
from datetime import datetime
from logging import getLogger, Logger
from threading import Thread, active_count
from tqdm import tqdm
from dataclasses import dataclass
from denspp.offline import get_path_to_project_start


# TODO: Add pipeline registry
class PipelineCMD:
    """Class for handling the pipeline processing"""
    path2save: str=''
    _path2pipe: str=''
    _path2start: str=get_path_to_project_start()
    _logger: Logger=getLogger(__name__)

    def get_pipeline_name(self) -> str:
        """Getting the name of the pipeline"""
        return self.__class__.__name__

    def generate_run_folder(self, path2runs: str, addon: str) -> None:
        """Generating the default folder for saving figures and data
        :param path2runs:   Main folder in which the figures and data is stored
        :param addon:       Name of new folder for saving results
        :return:            None
        """
        str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'{str_datum}_{self.get_pipeline_name().lower()}{addon}'
        path2start = join(self._path2start, path2runs)
        path2save = join(path2start, folder_name)
        makedirs(path2save, exist_ok=True)
        if self._path2pipe:
            copy(src=self._path2pipe, dst=path2save)
        self.path2save = path2save
        self._logger.debug(f"Creating run folder and copying the pipeline into: {path2save}")

    def apply_mapping(self, data: np.ndarray, electrode_id: list, mapping: np.ndarray) -> np.ndarray:
        """Transforming the input data to 2D array using electrode mapping configuration
        :param data:            Input data with shape (num_channels, num_samples)
        :param electrode_id:    List with name/numbers of electrodes used on data
        :param mapping:         Numpy array with electrode ID localisation
        :return:                Numpy array with transformed data to 2D
        """
        assert len(data.shape) == 2 and data.shape[0] > 1, "Shape of input data must higher than 2"
        assert data.shape[0] == len(electrode_id), "Mismatch between electrode_id and data shape"
        assert len(mapping), "No mapping is available"

        dut = np.zeros((mapping.shape[0], mapping.shape[1], data.shape[-1]), dtype=data.dtype)
        for row_idx in range(0, mapping.shape[0]):
            for col_idx in range(0, mapping.shape[1]):
                if mapping[row_idx, col_idx] > 0:
                    use_data_id = 0
                    for channel in electrode_id:
                        if channel == mapping[row_idx, col_idx]:
                            dut[row_idx, col_idx, :] = data[use_data_id, :]
                            break
                        use_data_id += 1
        self._logger.info("... transforming raw data array from 1D to 2D")
        return dut

    def deploy_mapping(self, data: np.ndarray, electrode_id: list, mapping: np.ndarray) -> np.ndarray:
        """Transforming the 2D data to normal electrode orientation using electrode mapping configuration
        :param data:            Input data with shape (num_rows, num_cols, num_samples)
        :param electrode_id:    List with name/numbers of electrodes used on data
        :param mapping:         Numpy array with electrode ID localisation
        :return:                Numpy array with original data format
        """
        assert len(data.shape) == 3, "Shape of input data must higher than 2"
        assert data.shape[0] * data.shape[1] >= len(electrode_id), "Mismatch between electrode_id and data shape"
        assert len(mapping), "No mapping is available"

        dut = np.zeros((len(electrode_id), data.shape[-1]), dtype=data.dtype)
        for row_idx in range(0, mapping.shape[0]):
            for col_idx in range(0, mapping.shape[1]):
                if mapping[row_idx, col_idx] > 0:
                    for channel in electrode_id:
                        if channel == mapping[row_idx, col_idx]:
                            dut[channel-1, :] = data[row_idx, col_idx, :]
                            break
        self._logger.info("... transforming raw data array from 2D to 1D")
        return dut

    def save_results(self, name: str, data: dict) -> None:
        """Saving the data with a dictionary
        :param name:    File name for saving results
        :param data:    Dictionary with data content
        :return:        None
        """
        path2data = join(self.path2save, name)
        np.save(path2data, data, allow_pickle=True)
        self._logger.info(f"... data saved in: {path2data}")


class ThreadProcessor(Thread):
    _logger: Logger
    output_save: dict
    _pipe_mode: int
    _num_jobs: int

    def __init__(self, rawdata: np.ndarray, fs_ana: float, pipeline) -> None:
        """Class for handling a thread of the signal processor with all dimensions
        :param rawdata:     Numpy array of input rawdata
        :param fs_ana:      Sampling rate of data [Hz]
        :param pipeline:    Pipeline/Thread Processor
        :return:            None
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self.input = rawdata
        self.pipeline = pipeline(fs_ana)
        self.output_save = dict()

    def run(self) -> None:
        """Do data processing"""
        self.output_save = self.pipeline.run(self.input)


@dataclass(frozen=True)
class SettingsThread:
    """Class for handling the processor
    Attribute:
        use_multithreading: Boolean for enabling multithreading on data processing pipeline
        num_max_workers:    Integer with total number of workers used in multithreading
        block_plots:        Boolean for blocking plotting
    """
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
            key = f'Elec_{elec}' if type(elec) == str else f'Elec_{elec:03d}'
            self.results_save.update({key: self.__threads_worker[idx].output_save})

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
                key = f'Elec_{elec}' if type(elec) == str else f'Elec_{elec:03d}'
                self.results_save.update({key: self.__threads_worker[idx].output_save})

    def do_save_results(self) -> None:
        """Saving results in desired numpy format"""
        path2save = self.__threads_worker[-1].pipeline.path2save
        np.save(f'{path2save}/results_half.npy', self.results_save)

    def do_plot_results(self) -> None:
        """Plotting the results of all signal processors"""
        plt.close('all')
        for key in self.results_save.keys():
            self.__threads_worker[-1].pipeline.do_plotting(self.results_save[key], int(key[-3:]))
        plt.show(block=self._settings.block_plots)

    def do_processing(self) -> None:
        """Performing the data processing"""
        if self._settings.use_multithreading and self._max_num_channel > 1:
            self.__perform_multi_threads()
        else:
            self.__perform_single_threads()
