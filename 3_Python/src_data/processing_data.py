import sys, os.path
import numpy as np
from datetime import date
from scipy.io import savemat
import matplotlib.pyplot as plt

from settings import Settings
from src.data_call import DataController
import src.amp.preamp
import src.adc.adc
import src.sda.sda

def cut_frames_from_stream(
        data: np.ndarray, xpos: int, dxneg: int, dxpos: int, cluster: np.ndarray, cluster_no: int
) -> [np.ndarray, np.ndarray]:
    frames_out = np.zeros(shape=(xpos.size, dxneg+dxpos))
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

def test_plot(frames_in, frames_cluster):
    cluster_no = np.unique(frames_cluster)

    frames_plot = []
    frames_mean = []
    for idx, clid in enumerate(cluster_no):
        xsel = np.where(frames_cluster == clid)
        frames0 = np.zeros(shape=(len(xsel[0]), frames_in.shape[1]))
        for i, sel in enumerate(xsel[0]):
            frames0[i, :] = frames_in[sel, :]

        frames_plot.append(frames0)
        frames_mean.append(np.mean(frames0, axis=0))

    # --- Plotting
    plt.figure()
    plt.plot(np.transpose(frames0), color='k')

    for idx, clid in enumerate(cluster_no):
        #plt.plot(np.transpose(frames_plot[idx]), color='k')
        plt.plot(frames_mean[idx])

    plt.show(block=True)

def get_frames_from_labeled_datasets(path2save: str, data_set: int, use_alldata: bool, align_mode: int, settings_afe: Settings, plot_result: bool):
    # --- Loading the pipeline
    preamp = src.amp.preamp.PreAmp(settings_afe, settings_afe.f_filt_ana)
    adc = src.adc.adc.ADC(settings_afe)
    sda = src.sda.sda.SDA(settings_afe)

    # --- Selection of datasets and points (Angabe in us)
    datahandler = DataController(settings_afe.path2data, 0)
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

        while (runPoint <= endPoint-1):
            # Calling the data
            datahandler.do_call(
                data_type=runSet,
                data_set=runPoint
            )
            datahandler.do_resample(
                t_range=np.array([]),
                desired_fs=settings_afe.fs_ana
            )
            data = datahandler.get_data()

            if data.label_exist == False:
                print("--- DATEIEN ENTHALTEN KEIN LABELING! ---")
                sys.exit()

            # Adapting the input for our application (Front-end emulation without filtering)
            u_ana = preamp.pre_amp(data.raw_data)
            x_adc = adc.adc_nyquist(u_ana, True)

            # Spike detection from labeling
            x_pos = np.floor(data.spike_xpos * settings_afe.fs_adc / data.fs_used).astype("int")
            x_start = np.floor(1e-6 * NegFrameExtension[runSet-1] / data.fs_used).astype("int")
            (frame_raw, frame_cluster) = cut_frames_from_stream(
                data=x_adc, xpos=x_pos,
                dxneg=x_start,
                dxpos=sda.frame_length + 2 * sda.offset_frame - x_start,
                cluster=data.cluster_id,
                cluster_no=frames_cluster
            )
            frame_aligned = sda.frame_aligning(frame_raw, align_mode)

            # Collecting datasets
            if first_run:
                frames_in = frame_aligned
                frames_cluster = frame_cluster
            else:
                frames_in = np.concatenate((frames_in, frame_aligned), axis=0)
                frames_cluster = np.concatenate((frames_cluster, frame_cluster), axis=0)

            first_run = False
            runPoint += 1
        # End of data point
        iteration += 1

    # --- Saving data
    create_time = date.today().strftime("%Y-%m-%d")
    newfile_name = os.path.join(path2save, (create_time + '_Dataset'))
    matdata = {"frames_in": frames_in, "frames_cluster": frames_cluster, "create_time": create_time, "settings": settings_afe}
    if use_alldata:
        newfile_name += '_Full'
    else:
        newfile_name += data.data_name

    savemat(newfile_name + '.mat', matdata)
    # np.savez(newfile_name + '.npz', frames_in, frames_cluster, create_time)
    print('\nSaving file in: ' + newfile_name + '.mat/.npz')

    # --- Plotting results
    test_plot(frames_in, frames_cluster)

    print("... This is the end")