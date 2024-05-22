import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, active_count
from tqdm import tqdm

from package.metric import Metric
from package.data_call.call_handler import SettingsDATA
from package.data_call.call_spike_files import DataLoader
from src_neuro.pipeline_v2 import Pipeline


SettingsDATA = SettingsDATA(
    path='../2_Data',
    data_set=1,
    data_case=0,
    data_point=0,
    t_range=[0],
    ch_sel=[],
    fs_resample=100e3
)


class CustomThread(Thread):
    output_full = None
    output_save = None

    def __init__(self, rawdata: np.ndarray):
        Thread.__init__(self)
        self.input = rawdata
        self.metric = Metric()
        self.pipeline = Pipeline(SettingsDATA.fs_resample)

    def run(self):
        self.pipeline.run(self.input)
        self.output_full = self.pipeline.signals
        self.output_save = self.pipeline.prepare_saving()


if __name__ == "__main__":
    block_plots = False
    use_multithreading = False
    print("\nRunning end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # ----- Preparation: Module calling -----
    datahand = DataLoader(SettingsDATA)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand

    # ----- Module declaration & Channel Calculation -----
    num_electrodes = dataIn.electrode_id
    pipe_test = Pipeline(SettingsDATA.fs_resample)
    path2save = pipe_test.path2save
    results_full = dict()
    results_save = dict()

    print("\nPerforming neural signal processing on all channels")
    if use_multithreading and len(num_electrodes) > 1:
        # --- Path for Multi-Threading
        max_num_workers = 2
        num_iterations = int(np.ceil(len(num_electrodes) / max_num_workers))
        process_threads = [None] * num_iterations
        for ite in range(num_iterations):
            process_threads[ite] = (num_electrodes[ite * max_num_workers: (ite + 1) * max_num_workers])

        print(f"... using {max_num_workers} of {active_count()} threading workers")
        for thr in tqdm(process_threads, ncols=100, desc='Progress: '):
            threads = list()
            for idx, elec in enumerate(thr):
                threads.append(CustomThread(dataIn.data_raw[elec]))
                threads[idx].start()

            for idx, elec in enumerate(thr):
                threads[idx].join()
                results_full[f'Elec_{elec:03d}'] = threads[idx].output_full
                results_save[f'Elec_{elec:03d}'] = threads[idx].output_save
    else:
        # --- Path for Single-Threading
        print('... using single threading')
        for idx, elec in enumerate(tqdm(num_electrodes, ncols=100, desc='Progress: ')):
            thread = CustomThread(dataIn.data_raw[idx])
            thread.start()
            thread.join()
            results_full[f'Elec_{elec:03d}'] = thread.output_full
            results_save[f'Elec_{elec:03d}'] = thread.output_save

    # --- Plot all plots and save results
    print("\nSaving results")
    pipe_test.save_results('results.mat', results_save)

    print("\nPlotting results")
    plt.close('all')
    for idx, elec in enumerate(num_electrodes):
        pipe_test.do_plotting(results_full[f"Elec_{elec:03d}"], elec)
    plt.show(block=block_plots)

    print("This is the End!")
