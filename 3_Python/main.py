import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, active_count
from tqdm import tqdm
from scipy.io import savemat

from package.metric import Metric
from package.data_call.call_spike_files import DataLoader, DataHandler, SettingsDATA
from src_neuro.pipeline_v1 import Pipeline


class ThreadProcessor(Thread):
    output_full = None
    output_save = None

    def __init__(self, rawdata: np.ndarray, fs_ana: float):
        Thread.__init__(self)
        self.input = rawdata
        self.metric = Metric()
        self.pipeline = Pipeline(fs_ana)

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


class ProcessingData:
    def __init__(self, settings: ThreadSettings, data_in: DataHandler):
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
            self.__threads_worker.append(ThreadProcessor(self.data.data_raw[idx], self._settings.fs_ana))
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
                self.__threads_worker.append(ThreadProcessor(dataIn.data_raw[elec], self._settings.fs_ana))
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


if __name__ == "__main__":
    # --- Settings for Processing
    SettingsDATA = SettingsDATA(
        # path='../2_Data',
        path='C:\HomeOffice\Data_Neurosignal',
        data_set=1,
        data_case=0,
        data_point=0,
        t_range=[0],
        ch_sel=[],
        fs_resample=100e3
    )
    settings_thr = ThreadSettings(
        use_multithreading=True,
        num_max_workers=2,
        block_plots=True,
        fs_ana=SettingsDATA.fs_resample
    )

    # ----- Preparation: Module calling -----
    print("Running framework for end-to-end neural signal processing (DeNSPP)"
          "\nStep #1: Loading data"
          "\n=================================================")
    datahand = DataLoader(SettingsDATA)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand

    # --- Thread Preparation: Processing data
    print("\nStep #2: Processing data"
          "\n=================================================")
    thr_station = ProcessingData(settings_thr, dataIn)
    thr_station.do_processing()

    # --- Plot all plots and save results
    print("\nStep #3: Saving results and plotting"
          "\n=================================================")
    thr_station.do_save_results()
    thr_station.do_save_results(True)
    thr_station.do_plot_results()
