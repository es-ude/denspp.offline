from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, active_count
from datetime import datetime
from scipy.io import savemat

from src_neuro.pipeline_v1 import Settings, Pipeline
from package.metric import Metric
from package.data.data_call import DataController
from package.plotting.plot_pipeline import results_afe1, results_afe2


class CustomThread(Thread):
    def __init__(self, rawdata: np.ndarray, channel: int, path2save=''):
        Thread.__init__(self)
        self.input = rawdata
        self.output = None
        self.out_save = dict()
        self.metric = None
        self.path2save = path2save
        self.thread_num = self.name
        self.channel = channel

    def run(self):
        print(f"... start processing of channel #{self.channel} on {self.thread_num}")
        # ---- Run src_neuro and calculate metrics
        SpikeSorting.run(self.input)
        self.metric = Metric()
        self.output = SpikeSorting.signals
        self.out_save = SpikeSorting.prepare_saving()
        print(f"... process done ({self.thread_num} closed)")


def save_results(data: list, path2save="") -> None:
    """Function for plotting the results"""
    name = 'results.mat'
    mdict = dict()
    for elec, val in enumerate(data):
        mdict0 = {'frames_out': val.frames_align[0],
                  'frames_pos': val.frames_align[1],
                  'frames_clu': val.frames_align[2],
                  'fs_dig': val.fs_dig}
        mdict.update({f'Elec_{elec:03d}': mdict0})
    savemat(join(path2save, name), mdict)


def func_plots(data, channel: int, path2save="") -> None:
    """Function to plot resilts from spike sorting"""
    # --- Spike Sorting output
    results_afe1(data, channel, path=path2save)
    results_afe2(data, channel, path=path2save)
    #results_afe2(data, channel, path=path2save, time_cut=[10, 12])
    # results_fec(data, channel, path=path2save)
    # results_paper(data, channel, path=path2save)

    # --- NSP block
    # results_ivt(data, channel, path=path2save)
    # results_firing_rate(data, channel, path=path2save)
    # results_correlogram(data, channel, path=path2save)
    # results_cluster_amplitude(data, channel, path=path2save)


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
    num_electrodes = dataIn.electrode_id[0]
    SpikeSorting = Pipeline(settings)
    path2save = SpikeSorting.generate_folder(folder_name)
    results = [None] * len(num_electrodes)

    print("\nPerforming end-to-end src_neuro on all channels")
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
                threads.append(CustomThread(dataIn.data_raw[elec], elec, path2save))
                threads[idx].start()

            for idx, elec in enumerate(thr):
                threads[idx].join()
                results[elec] = threads[idx].output
    else:
        # --- Path for Single-Threading
        for idx, elec in enumerate(num_electrodes):
            thread = CustomThread(dataIn.data_raw[idx], elec, path2save)
            thread.start()
            thread.join()
            results[elec] = thread.output

    # --- Plot all plots and save results (must be externally)
    print("\nSaving and plotting the results")
    save_results(results, path2save=path2save)
    for idx, elec in enumerate(num_electrodes):
        func_plots(results[elec], elec, path2save)

    print("This is the End!")
    plt.show(block=True)
