from glob import glob
from os.path import join, exists
from os import mkdir
import numpy as np
from datetime import datetime
from scipy.io import savemat, loadmat
from tqdm import tqdm

from package.data.data_call import DataController
from src_data.pipeline_data import Settings, Pipeline


def get_frames_from_dataset(path2save: str, cluster_class_avai=False, process_points=[]) -> None:
    """
    Tool for loading datasets in order to generate one new dataset (Step 1),
    cluster_class_avai: False = Concatenate the class number with increasing id number (useful for non-biological clusters)
    only_pos: Taking the datapoints of the choicen dataset [Start, End]
    """
    # --- Loading the src_neuro
    afe_set = Settings()
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # ------ Loading Data: Preparing Data
    create_time = datetime.now().strftime("%Y-%m-%d")
    print("... loading the datasets")
    path2folder = join(path2save, 'Merging')

    if not exists(path2folder):
        mkdir(path2folder)

    # --- Calling the data into RAM
    settings = dict()
    runPoint = process_points[0] if len(process_points) > 0 else 0
    endPoint = 0

    first_run = True
    while first_run or runPoint < endPoint:
        first_run = True
        timepoint_start = datetime.now()

        frames_in = np.empty(shape=(0, 0), dtype=np.dtype('int16'))
        frames_cluster = np.empty(shape=(0, 0), dtype=np.dtype('uint16'))

        afe_set.SettingsDATA.data_point = runPoint
        datahandler = DataController(afe_set.SettingsDATA)
        datahandler.do_call()
        datahandler.do_resample()

        # --- Taking signals from handler
        for ch in tqdm(datahandler.raw_data.electrode_id, ncols=100, desc="Progress: "):
            cl_in = datahandler.raw_data.cluster_id[ch]
            spike_xpos = np.floor(datahandler.raw_data.spike_xpos[ch] * fs_adc / fs_ana).astype("int")
            spike_xoff = int(1e-6 * datahandler.raw_data.spike_offset_us[0] * fs_adc)

            # --- Processing the analogue input
            afe = Pipeline(afe_set)
            afe.run_input(datahandler.raw_data.data_raw[ch], spike_xpos, spike_xoff)

            # --- Post-Processing: Checking if same length
            if afe.signals.frames_align.shape[0] != spike_xpos.size:
                continue

            # --- Processing (Frames and cluster)
            max_cluster_num = 0 if (first_run or cluster_class_avai) else (1 + np.argmax(np.unique(frames_cluster)))
            if first_run:
                endPoint = process_points[1] if len(process_points) == 2 else datahandler.no_files
                settings = afe.save_settings()
                frames_in = afe.signals.frames_align
                frames_cluster = cl_in + max_cluster_num
            else:
                frames_in = np.concatenate((frames_in, afe.signals.frames_align), axis=0)
                frames_cluster = np.concatenate((frames_cluster, cl_in + max_cluster_num), axis=0)
            first_run = False

            # --- Release memory
            del afe, spike_xpos, cl_in

        print(f"... done after {1e-6 * (datetime.now() - timepoint_start).microseconds: .2f} s")
        # --- Saving data (each run)
        newfile_name = join(path2folder, (create_time + '_Dataset-'
                                          + datahandler.raw_data.data_name
                                          + f'_step{runPoint + 1:03d}'))
        savemat(newfile_name + '.mat', {"frames_in": frames_in,
                   "frames_cluster": frames_cluster,
                   "create_time": create_time, "settings": settings})
        print('Saving file in: ' + newfile_name + '.mat')

        # --- Release memory
        del datahandler, frames_in, frames_cluster

        # --- End control routine
        runPoint += 1

    # --- The End
    print("... This is the end")


def merge_frames_from_dataset() -> None:
    """Tool for merging all spike frames to one new dataset (Step 2)"""
    print("\nStart MATLAB script manually: merge/merge_datasets_matlab.m")


def merge_data_from_diff_data(path2data: str) -> None:
    folder_content = glob(join(path2data, 'Merging', '*.mat'))
    folder_content.sort()

    frame_in = list()
    frame_cl = list()

    for idx, file in enumerate(folder_content):
        print(idx, file)
        data = loadmat(file)

        frame_in = data['frames_in'] if idx == 0 else np.append(frame_in, data['frames_in'], axis=0)
        frame_cl = data['frames_cluster'] if idx == 0 else np.append(frame_cl, data['frames_cluster'], axis=0)

        if idx == 0:
            file_name = file

    newfile_name = join(path2data, file_name)
    savemat(newfile_name + '.mat', {"frames_in": frame_in,
                                    "frames_cluster": frame_cl,
                                    "create_time": data['create_time'], "settings": data['settings']})
