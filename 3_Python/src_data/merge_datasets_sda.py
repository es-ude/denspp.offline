from os.path import join
import numpy as np
from time import time_ns
from scipy.io import savemat
from tqdm import tqdm

from package.data.data_call import DataController
from src_data.pipeline_data import Settings, Pipeline


def prepare_sda_dataset(path2save: str, process_points=[]) -> None:
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
    datahandler = DataController(afe_set.SettingsDATA)
    datahandler.do_call()
    datahandler.do_resample()
    data = datahandler.get_data()
    del datahandler

    # --- Taking signals from handler
    sda_input = list()
    sda_pred = list()
    xpos = None
    for idx, elec in enumerate(tqdm(data.electrode_id, ncols=100, desc='Progress:')):
        spike_xpos = np.floor(data.spike_xpos[elec] * fs_adc / fs_ana).astype("int")
        spike_xoff = int(1e-6 * data.spike_offset_us[0] * fs_adc)

        pipeline = Pipeline(afe_set)
        pipeline.run_input(data.data_raw[elec], spike_xpos, spike_xoff)

        # --- Process dataset for SDA
        len_data = int(pipeline.signals.x_adc.shape[0] / 8)
        xpos = pipeline.signals.x_pos
        sda_input.append(pipeline.signals.x_adc.reshape(len_data, 8))
        sda_pred.append(np.zeros(shape=sda_input[idx].shape, dtype=bool))

        # --- Release memory
        del pipeline

    # --- Saving results

    mdict = {'fs_adc': fs_adc,
             'sda_in': sda_input[0],
             'sda_pred': sda_pred[0],
             'sda_xpos': xpos}
    newfile_name = join(path2save, 'SDA_Dataset' + '.mat')
    savemat(newfile_name, mdict)
    print('... saving results in: ' + newfile_name)

    delta_time = 1e-9 * (time_ns() - timepoint_start)
    print(f"... done after {delta_time: .4f} s")
