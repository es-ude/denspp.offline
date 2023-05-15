import sys, os.path
import numpy as np
from datetime import date
from scipy.io import savemat
import matplotlib.pyplot as plt

from src.data_call import DataController
from pipeline.pipeline_data import Settings, Pipeline

def cut_frames_from_stream(
        data: np.ndarray, xpos: np.ndarray, dx: [int, int], cluster: np.ndarray, cluster_no: np.ndarray
) -> [np.ndarray, np.ndarray]:
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

def get_frames_from_labeled_datasets(path2save: str, data_set: int, use_alldata: bool, align_mode: int):
    # --- Loading the pipeline
    afe_set = Settings()
    afe = Pipeline(afe_set)
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # --- Selection of datasets and points (Angabe in us)
    datahandler = DataController(afe_set.SettingsDATA)
    MaxDataPoints = datahandler.max_datapoints
    NegFrameExtension = np.array([100, 100, 3, 3], dtype="int8")
    if use_alldata:
        TakeDatasets = np.arange(1, MaxDataPoints.size)
        # If value = 0 --> All data points will be loaded - otherwise Maximum number
        NoDataPoints = np.array([0, 0, 0, 0])
    else:
        TakeDatasets = np.array([data_set])
        NoDataPoints = np.array([5])
    # ------ Loading Data: Preparing Data
    print("... loading the datasets")
    frames_in = np.empty(shape=(0, 0), dtype=int)
    frames_cluster = np.empty(shape=(0, 0), dtype=int)
    # ------ Loading Data: Running
    iteration = 0
    first_run = True
    for runSet in TakeDatasets:
        runPoint = 0
        if NoDataPoints[iteration] == 0:
            endPoint = MaxDataPoints[runSet-1]
        else:
            endPoint = NoDataPoints[iteration]

        # --- Calling the data into RAM
        while (runPoint <= endPoint-1):
            afe_set.SettingsDATA.data_point = runPoint
            datahandler.do_call()
            datahandler.do_resample()
            data = datahandler.get_data()
            # --- Taking signals from handler
            u_in = data.raw_data[0]
            cl_in = data.cluster_id[0]
            x_pos = data.spike_xpos[0]
            # --- Apply analogue signal into pipeline
            afe.run_input(u_in)
            # --- Spike Detection from Labeling
            x_pos = np.floor(x_pos * fs_adc / fs_ana).astype("int")
            x_start = np.floor(1e-6 * NegFrameExtension[runSet - 1] / fs_ana).astype("int")
            (frame_raw, frame_cluster) = cut_frames_from_stream(
                data=afe.x_adc, xpos=x_pos,
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
        # End of data point
        iteration += 1

    # --- Saving data
    create_time = date.today().strftime("%Y-%m-%d")
    newfile_name = os.path.join(path2save, (create_time + '_Dataset'))
    matdata = {"frames_in": frames_in, "frames_cluster": frames_cluster, "create_time": create_time, "settings": afe_set}
    if use_alldata:
        newfile_name += '_Full'
    else:
        newfile_name += data.data_name

    savemat(newfile_name + '.mat', matdata)
    # np.savez(newfile_name + '.npz', frames_in, frames_cluster, create_time)
    print('\nSaving file in: ' + newfile_name + '.mat/.npz')
    print("... This is the end")