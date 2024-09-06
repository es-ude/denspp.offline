from os.path import join
import numpy as np
from time import time_ns
from datetime import datetime
from scipy.io import savemat
from tqdm import tqdm

from package.data_call.call_handler import _DataController
from src_neuro.pipeline_data import Settings, Pipeline


def prepare_sda_dataset(path2save: str, slice_size=12, process_points=[]) -> None:
    """Tool for loading datasets in order to generate one new dataset (Step 1),
        cluster_class_avai: False = Concatenate the class number with increasing id number (useful for non-biological clusters)
        only_pos: Taking the datapoints of the choicen dataset [Start, End]"""
    # --- Loading the src_neuro
    afe_set = Settings()
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # ------ Loading Data: Preparing Data
    timepoint_start = time_ns()
    print("\nStart merging datasets for generating a dataset for train spike detection algorithms (SDA)")
    print(f"... loading the datasets")
    datahandler = _DataController(afe_set.SettingsDATA)
    datahandler.do_call()
    datahandler.do_resample()
    data = datahandler.get_data()
    del datahandler

    # --- Taking signals from handler
    sda_input = list()
    sda_pred = list()
    xpos = None
    for idx0, elec in enumerate(tqdm(data.electrode_id, ncols=100, desc='Progress:')):
        window = [-16, 40]
        spike_clus = data.evnt_cluster_id[elec]
        pos = np.argwhere(spike_clus != 0)
        spike_xpos = np.floor(data.evnt_xpos[elec][pos].flatten() * fs_adc / fs_ana).astype("int")
        spike_xoff = int(1e-6 * data.spike_offset_us[0] * fs_adc)

        pipeline = Pipeline(afe_set)
        pipeline.run_input(data.data_raw[elec], spike_xpos, spike_xoff)

        # --- Process dataset for SDA
        cut_end = (pipeline.signals.x_adc.size % slice_size)
        data_in = pipeline.signals.x_adc if cut_end == 0 else pipeline.signals.x_adc[:-cut_end]

        data_sda = np.zeros(shape=data_in.shape, dtype=bool)
        for xpos in (spike_xpos + spike_xoff + 25):
            for dx in range(window[0], window[1]):
                data_sda[int(xpos+dx)] = True

        # --- Slicing the data
        sda_input.append(np.array(data_in.reshape((int(data_in.size/slice_size), slice_size)), dtype=int))
        sda_pred.append(np.array(data_sda.reshape((int(data_sda.size/slice_size), slice_size)), dtype=bool))

        # --- Release memory
        del pipeline, data_in, data_sda

    # --- Saving results
    create_time = datetime.now().strftime("%Y-%m-%d")
    mdict = {'fs_adc': fs_adc,
             'sda_in': sda_input[0],
             'sda_pred': sda_pred[0],
             'sda_xpos': xpos}
    newfile_name = join(path2save, create_time + '_SDA_Dataset' + '.mat')
    savemat(newfile_name, mdict)
    print('... saving results in: ' + newfile_name)

    delta_time = 1e-9 * (time_ns() - timepoint_start)
    print(f"... done after {delta_time: .4f} s")
