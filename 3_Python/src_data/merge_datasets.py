import os.path
import numpy as np
from datetime import date
from scipy.io import savemat

from src.data_call import DataController
from pipeline.pipeline_data import Settings, Pipeline

def cut_frames_from_stream(
        data: np.ndarray, xpos: np.ndarray, dx: [int, int], cluster: np.ndarray, cluster_no: np.ndarray
) -> [np.ndarray, np.ndarray]:
    """Tool for cutting the spike frames out of the bitstream
    (only useable with already labeled information)"""
    dxneg = dx[0]
    dxpos = dx[1]
    frames_out = np.zeros(shape=(xpos.size, dxneg + dxpos))
    frames_cluster = np.zeros(shape=(xpos.size, 1))

    if cluster_no.size == 0:
        max_val = 0
    else:
        max_val = 1 + np.argmax(np.unique(cluster_no))

    idx = 0
    for pos in xpos:
        frames_out[idx, :] = data[pos-dxneg:pos+dxpos]
        frames_cluster[idx] = max_val + cluster[idx]
        idx += 1

    return frames_out, frames_cluster

def get_frames_from_dataset(path2save: str, data_set: int, align_mode: int):
    """Tool for loading datasets in order to generate one new dataset (Step 1)"""
    # --- Loading the pipeline
    afe_set = Settings()
    afe = Pipeline(afe_set)
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # ------ Loading Data: Preparing Data
    print("... loading the datasets")
    afe_set.SettingsDATA.data_set = data_set
    datahandler = DataController(afe_set.SettingsDATA)

    frames_in = np.empty(shape=(0, 0), dtype=int)
    frames_cluster = np.empty(shape=(0, 0), dtype=int)

    # --- Calling the data into RAM
    runPoint = 0
    endPoint = 0
    first_run = True
    file_name = "Test"
    while first_run or runPoint < endPoint:
        afe_set.SettingsDATA.data_point = runPoint
        datahandler.do_call()
        datahandler.do_resample()
        data = datahandler.get_data()
        # --- Taking signals from handler
        u_in = data.raw_data[0]
        cl_in = data.cluster_id[0]
        spike_xpos = data.spike_xpos[0]
        spike_offset = data.spike_offset[0]
        file_name = data.data_name
        endPoint = datahandler.dataset_numpoints
        # --- Apply analogue signal into pipeline
        afe.run_input(u_in)
        # --- Spike Detection from Labeling
        spike_xpos = np.floor(spike_xpos * fs_adc / fs_ana).astype("int")
        x_start = np.floor(1e-6 * spike_offset / fs_ana).astype("int")
        (frame_raw, frame_cluster) = cut_frames_from_stream(
            data=afe.x_adc, xpos=spike_xpos,
            dx=[x_start, afe.sda.offset_frame + afe.sda.frame_length - x_start],
            cluster=cl_in,
            cluster_no=frames_cluster
        )
        frame_aligned = afe.sda.frame_aligning(frame_raw, align_mode)
        # --- Collecting datasets
        if first_run:
            frames_in = frame_aligned
            frames_cluster = frame_cluster
        else:
            frames_cluster = np.concatenate((frames_cluster, frame_cluster), axis=0)
            frames_in = np.concatenate((frames_in, frame_aligned), axis=0)

        first_run = False
        runPoint += 1

    # --- Saving data
    create_time = date.today().strftime("%Y-%m-%d")
    newfile_name = os.path.join(path2save, (create_time + '_Dataset' + file_name))
    matdata = {"frames_in": frames_in, "frames_cluster": frames_cluster, "create_time": create_time, "settings": afe_set}

    savemat(newfile_name + '.mat', matdata)
    # np.savez(newfile_name + '.npz', frames_in, frames_cluster, create_time)
    print('\nSaving file in: ' + newfile_name + '.mat/.npz')
    print("... This is the end")

def merge_frames_from_dataset():
    """Tool for merging all spike frames to one new dataset (Step 2)"""
    print("... Start MATLAB script manually: src_data/merge_datasets_matlab.m")