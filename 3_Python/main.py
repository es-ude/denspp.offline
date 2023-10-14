import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, active_count
from datetime import datetime

from pipeline.pipeline_v1 import Settings, Pipeline
from package.metric import Metric
from package.data_call import DataController
from package.plotting import results_afe1, results_fec, results_paper, results_ivt, results_firing_rate, results_correlogram


class CustomThread(Thread):
    def __init__(self, thread_num: int, rawdata: np.ndarray, channel: int, path2save=''):
        Thread.__init__(self)
        self.input = rawdata
        self.output = None
        self.metric = None
        self.path2save = path2save
        self.thread_num = self.name
        self.channel = channel

    def run(self):
        print(f"... start processing of channel #{self.channel} on {self.thread_num}")
        # ---- Run pipeline and calculate metrics
        SpikeSorting.run(self.input)
        self.metric = Metric()
        self.output = SpikeSorting.signals
        SpikeSorting.run_nsp()
        SpikeSorting.saving_mat(self.channel)
        print(f"... process done ({self.thread_num} closed)")


def func_plots(data, channel: int, path2save="") -> None:
    """Function to plot resilts from spike sorting"""
    results_afe1(data, channel, path=path2save)
    results_fec(data, channel, path=path2save)
    # results_paper(data, path2save, channel, path=path2save)
    # results_ivt(data, path2save, channel, path=path2save)
    results_firing_rate(data, channel, path=path2save)
    results_correlogram(data, channel, path=path2save)


if __name__ == "__main__":
    plt.close('all')

    use_multithreading = False
    str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = '{}_pipeline'.format(str_datum)
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # ----- Preparation: Module calling -----
    settings = Settings()
    datahand = DataController(settings.SettingsDATA)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()

    # ----- Module declaration & Channel Calculation -----
    num_electrodes = dataIn.channel
    SpikeSorting = Pipeline(settings)
    path2save = SpikeSorting.generate_folder(folder_name)
    results = [None] * len(num_electrodes)

    print("\nPerforming end-to-end pipeline on all channels")
    if use_multithreading and len(num_electrodes) > 1:
        # --- Path for Multi-Threading
        max_num_workers = 2
        print(f"... using {max_num_workers} of {active_count()} threading workers")
        num_iterations = int(np.ceil(len(num_electrodes) / max_num_workers))
        process_threads = [None] * num_iterations
        for ite in range(num_iterations):
            process_threads[ite] = (num_electrodes[ite * max_num_workers: (ite + 1) * max_num_workers])

        for thr in process_threads:
            threads = list()
            for idx, elec in enumerate(thr):
                threads.append(CustomThread(idx, dataIn.raw_data[elec], elec, path2save))
                threads[idx].start()

            for idx, elec in enumerate(thr):
                threads[idx].join()
                results[elec] = threads[idx].output
    else:
        # --- Path for Single-Threading
        for idx, elec in enumerate(num_electrodes):
            thread = CustomThread(0, dataIn.raw_data[idx], elec, path2save)
            thread.start()
            thread.join()
            results[elec] = thread.output

    # --- Plot all plots and save results (must be externally)
    for idx, elec in enumerate(num_electrodes):
        func_plots(results[elec], elec, path2save)

    plt.show(block=False)
