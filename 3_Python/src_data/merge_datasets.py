import os.path
import numpy as np
from datetime import date
from scipy.io import savemat
from tqdm import tqdm

from package.data.data_call import DataController
from src_data.pipeline_data import Settings, Pipeline


def get_frames_from_dataset(path2save: str, cluster_class_avai=False, process_points=[0], do_step_save=False):
    """Tool for loading datasets in order to generate one new dataset (Step 1),
    cluster_class_avai: False = Concatenate the class number with increasing id number (useful for non-biological clusters)
    only_pos: Taking the datapoints of the choicen dataset [Start, End]"""
    # --- Loading the src_neuro
    afe_set = Settings()
    afe = Pipeline(afe_set)
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # ------ Loading Data: Preparing Data
    print("... loading the datasets")
    datahandler = DataController(afe_set.SettingsDATA)

    frames_in = np.empty(shape=(0, 0), dtype=np.dtype('int16'))
    frames_cluster = np.empty(shape=(0, 0), dtype=np.dtype('uint16'))

    # --- Calling the data into RAM
    runPoint = process_points[0]
    endPoint = 0
    first_run = True
    while first_run or runPoint < endPoint:
        afe_set.SettingsDATA.data_point = runPoint
        datahandler.do_call()
        datahandler.do_resample()
        data = datahandler.get_data()

        # --- Taking signals from handler
        for ch in tqdm(data.electrode_id, ncols=100, desc="Progress: "):
            u_in = data.data_raw[ch]
            cl_in = data.cluster_id[ch]
            spike_xpos = data.spike_xpos[ch]
            spike_offset = data.spike_offset[0]

            # --- Pre-Processing with analogue src_neuro
            afe.run_input(u_in)
            spike_xpos = np.floor(spike_xpos * fs_adc / fs_ana).astype("int")
            x_start = np.floor(1e-6 * spike_offset / fs_ana).astype("int")

            # --- Processing (Frames and cluster)
            frame_aligned = afe.sda.frame_generation_pos(afe.x_adc, spike_xpos, x_start)[1]

            max_cluster_num = 0 if first_run else 1 + np.argmax(np.unique(frames_cluster))
            new_cluster_add = cl_in if cluster_class_avai else (max_cluster_num + cl_in)
            if first_run:
                endPoint = datahandler.no_files if len(process_points) == 1 else process_points[1]
                frames_in = frame_aligned
                frames_cluster = new_cluster_add
            else:
                # endPoint = endPoint
                frames_cluster = np.concatenate((frames_cluster, new_cluster_add), axis=0)
                frames_in = np.concatenate((frames_in, frame_aligned), axis=0)
            first_run = False

        # --- End control routine
        file_name = data.data_name
        if do_step_save:
            create_time = date.today().strftime("%Y-%m-%d")
            newfile_name = os.path.join(path2save, (create_time + f'_Dataset_Step{runPoint:03d}' + file_name))
            matdata = {"frames_in": frames_in,
                       "frames_cluster": frames_cluster,
                       "create_time": create_time, "settings": afe_set}
            savemat(newfile_name + '.mat', matdata)
        runPoint += 1

    # --- Saving data
    create_time = date.today().strftime("%Y-%m-%d")
    newfile_name = os.path.join(path2save, (create_time + '_Dataset' + file_name))
    matdata = {"frames_in": frames_in,
               "frames_cluster": frames_cluster,
               "create_time": create_time, "settings": afe_set}
    savemat(newfile_name + '.mat', matdata)
    # np.savez(newfile_name + '.npz', frames_in, frames_cluster, create_time)
    print('\nSaving file in: ' + newfile_name + '.mat/.npz')
    print("... This is the end")


def merge_frames_from_dataset():
    """Tool for merging all spike frames to one new dataset (Step 2)"""
    print("... Start MATLAB script manually: merge/merge_datasets_matlab.m")