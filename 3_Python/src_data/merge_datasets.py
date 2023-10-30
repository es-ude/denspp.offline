import os.path
import numpy as np
from datetime import date
from scipy.io import savemat
from tqdm import tqdm
from gc import collect

from package.data.data_call import DataController
from src_data.pipeline_data import Settings, Pipeline


def get_frames_from_dataset_common(path2save: str, cluster_class_avai=False, process_points=[]):
    """Tool for loading datasets in order to generate one new dataset (Step 1),
    cluster_class_avai: False = Concatenate the class number with increasing id number (useful for non-biological clusters)
    only_pos: Taking the datapoints of the choicen dataset [Start, End]"""
    # --- Loading the src_neuro
    afe_set = Settings()
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # ------ Loading Data: Preparing Data
    print("... loading the datasets")

    frames_in = np.empty(shape=(0, 0), dtype=np.dtype('int16'))
    frames_cluster = np.empty(shape=(0, 0), dtype=np.dtype('uint16'))

    # --- Calling the data into RAM
    runPoint = process_points[0]
    endPoint = 0
    first_run = True
    file_name = ''
    settings = dict()
    while first_run or runPoint < endPoint:
        afe_set.SettingsDATA.data_point = runPoint
        datahandler = DataController(afe_set.SettingsDATA)
        datahandler.do_call()
        datahandler.do_resample()
        data = datahandler.get_data()
        endPoint = datahandler.no_files if len(process_points) == 1 else process_points[1]
        del datahandler

        # --- Taking signals from handler
        for ch in tqdm(data.electrode_id, ncols=100, desc="Progress: "):
            u_in = data.data_raw[ch]
            cl_in = data.cluster_id[ch]
            spike_xpos_orig = data.spike_xpos[ch]
            spike_offset = int(1e-6 * data.spike_offset_us[0] * fs_adc)

            # --- Pre-Processing with analogue src_neuro
            afe = Pipeline(afe_set)
            afe.run_input(u_in)
            spike_xpos_adc = np.floor(spike_xpos_orig * fs_adc / fs_ana).astype("int")

            # --- Processing (Frames and cluster)
            frame_aligned = afe.sda.frame_generation_pos(afe.x_adc, spike_xpos_adc, spike_offset)[1]
            max_cluster_num = 0 if (first_run or cluster_class_avai) else 1 + np.argmax(np.unique(frames_cluster))
            if first_run:
                settings = afe.save_settings()
                frames_in = frame_aligned
                frames_cluster = cl_in + max_cluster_num
            else:
                frames_in = np.concatenate((frames_in, frame_aligned), axis=0)
                frames_cluster = np.concatenate((frames_cluster, cl_in + max_cluster_num), axis=0)
            first_run = False

            # --- Release memory
            del afe
            del u_in, cl_in, frame_aligned
            del spike_xpos_adc, spike_xpos_orig
        file_name = data.data_name
        del data

        # --- End control routine
        runPoint += 1

    # --- Saving data
    create_time = date.today().strftime("%Y-%m-%d")
    matdata = {"frames_in": frames_in,
               "frames_cluster": frames_cluster,
               "create_time": create_time, "settings": settings}
    newfile_name = os.path.join(path2save, (create_time + '_Dataset-' + file_name))
    savemat(newfile_name + '.mat', matdata)
    print('\nSaving file in: ' + newfile_name + '.mat/.npz')
    print("... This is the end")


def get_frames_from_dataset_unique(path2save: str, cluster_class_avai=False, process_points=[]):
    """Tool for loading datasets in order to generate one new dataset (Step 1),
    cluster_class_avai: False = Concatenate the class number with increasing id number (useful for non-biological clusters)
    only_pos: Taking the datapoints of the choicen dataset [Start, End]"""
    # --- Loading the src_neuro
    afe_set = Settings()
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # ------ Loading Data: Preparing Data
    print("... loading the datasets")
    runPoint = process_points[0]
    endPoint = 0
    first_run = True
    settings = dict()
    while first_run or runPoint < endPoint:
        first_run = True
        frames_in = np.zeros(shape=(0, 0), dtype=np.dtype('int16'))
        frames_cluster = np.zeros(shape=(0, 0), dtype=np.dtype('uint16'))

        afe_set.SettingsDATA.data_point = runPoint
        datahandler = DataController(afe_set.SettingsDATA)
        datahandler.do_call()
        datahandler.do_resample()
        data = datahandler.get_data()
        endPoint = datahandler.no_files if len(process_points) == 1 else process_points[1]
        del datahandler

        # --- Taking signals from handler
        for ch in tqdm(data.electrode_id, ncols=100, desc="Progress: "):
            u_in = data.data_raw[ch]
            cl_in = data.cluster_id[ch]
            spike_xpos_orig = data.spike_xpos[ch]
            spike_offset = int(1e-6 * data.spike_offset_us[0] * fs_adc)

            # --- Pre-Processing with analogue src_neuro
            afe = Pipeline(afe_set)
            afe.run_input(u_in)
            spike_xpos_adc = np.floor(spike_xpos_orig * fs_adc / fs_ana).astype("int")

            # --- Processing (Frames and cluster)
            frame_aligned = afe.sda.frame_generation_pos(afe.x_adc, spike_xpos_adc, spike_offset)[1]
            max_cluster_num = 0 if (first_run or cluster_class_avai) else 1 + np.argmax(np.unique(frames_cluster))
            if first_run:
                settings = afe.save_settings()
                frames_in = frame_aligned
                frames_cluster = cl_in + max_cluster_num
            else:
                frames_in = np.concatenate((frames_in, frame_aligned), axis=0)
                frames_cluster = np.concatenate((frames_cluster, cl_in + max_cluster_num), axis=0)
            first_run = False

            # --- delete large variables and release memory
            del afe
            del u_in, cl_in, spike_xpos_orig, frame_aligned
            del spike_xpos_adc, spike_xpos_orig

        file_name = data.data_name
        del data

        # --- End control routine
        create_time = date.today().strftime("%Y-%m-%d")
        matdata = {"frames_in": frames_in,
                   "frames_cluster": frames_cluster,
                   "create_time": create_time, "settings": settings}
        newfile_name = os.path.join(path2save, (create_time + f'_Dataset-' + file_name) + f'-Step{runPoint:03d}')
        savemat(newfile_name + '.mat', matdata)
        runPoint += 1
    print("... This is the end")


def merge_frames_from_dataset():
    """Tool for merging all spike frames to one new dataset (Step 2)"""
    print("... Start MATLAB script manually: merge/merge_datasets_matlab.m")