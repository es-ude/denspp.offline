import numpy as np

from settings import Settings
from src.call_data import call_data
from src.afe import AFE
from scipy.io import savemat

def getting_frames(data: np.ndarray, xpos: int, dxneg: int, dxpos: int, cluster: np.ndarray, cluster_no: int):
    frames_out = np.zeros(shape=(xpos.size, dxneg+dxpos))
    frames_cluster = np.zeros(shape=(xpos.size, 1))

    if cluster_no.size == 0:
        max_val = 0
    else:
        max_val = 1+np.argmax(np.unique(cluster_no))

    idx = 0
    for pos in xpos:
        frames_out[idx, :] = data[pos-dxneg:pos+dxpos]
        frames_cluster[idx] = max_val + cluster[idx]
        idx += 1

    return frames_out, frames_cluster


def LoadSpAIke_Data(path2file: str, use_fulldata: bool, align_mode: int, separate_files: bool):
    settings = Settings()
    afe = AFE(settings)

    # --- Selection of datasets and points
    MaxDataPoints = np.array([5, 16, 22])
    SamplingPoints = np.array([[5, 4, 3], [15, 14, 13]], dtype="int8")
    if use_fulldata:
        TakeDatasets = np.array([1, 2, 3])
        # If value = 0 --> All data points will be loaded - otherwise Maximum number
        NoDataPoints = np.array([0, 0, 0])
    else:
        TakeDatasets = np.array([1])
        NoDataPoints = np.array([0])

    # ------ Loading Data: Preparing Data
    print("... loading the datasets")
    frames_in = np.empty(shape=(0, 0), dtype=int)
    frames_cluster = np.empty(shape=(0, 0), dtype=int)
    runs = 0

    # TODO: Adding meta-informations

    # ------ Loading Data: Running
    iteration = 0
    for runSet in TakeDatasets:
        runPoint = 1
        SamplingRange = SamplingPoints[:, runSet - 1]

        if NoDataPoints[iteration] == 0:
            endPoint = MaxDataPoints[runSet-1]

        while (runPoint <= endPoint):
            # Calling the data
            (neuron, labeling) = call_data(
                path2data=settings.path2data,
                data_type=runSet,
                data_set=runPoint,
                desired_fs=settings.desired_fs,
                t_range=settings.t_range,
                ch_sel=settings.ch_sel,
                plot=False
            )

            # Adapting the input for our application (Front-end emulation without filtering)
            u_ana, _ = afe.pre_amp(neuron.data)
            x_adc = afe.adc_nyquist(u_ana, 1)

            # Spike detection from labeling
            x_pos = np.floor(labeling.spike_xpos * settings.sample_rate / neuron.fs).astype("int")
            (frame_raw, frame_cluster) = getting_frames(
                data=x_adc, xpos=x_pos,
                dxneg=SamplingRange[0], dxpos=SamplingRange[1] + afe.frame_length,
                cluster=labeling.cluster_id,
                cluster_no=frames_cluster
            )
            # Aligning all frames
            if(0):
                frame_aligned = frame_raw
            else:
                frame_aligned = afe.frame_aligning(frame_raw, align_mode, 1)
            # Collecting datasets
            if runs == 0:
                frames_in = frame_aligned
                frames_cluster = frame_cluster
            else:
                frames_in = np.concatenate((frames_in, frame_aligned), axis=0)
                frames_cluster = np.concatenate((frames_cluster, frame_cluster), axis=0)

            runs += 1
            runPoint += 1

        iteration += 1

        if separate_files:
            matdata = {"frames_in": frames_in, "frames_cluster": frames_cluster}
            print('Saving file in: ' + path2file + '_File' + runSet.astype('str') + '.mat')
            savemat(path2file + '_File' + runSet.astype('str') + '.mat', matdata)
            # np.savez(path2file + '.npz', frames_in, frames_cluster)
            runs = 0
            frames_in = np.empty(shape=(0, 0), dtype=int)
            frames_cluster = np.empty(shape=(0, 0), dtype=int)

    # --- Saving data
    if not separate_files:
        matdata = {"frames_in": frames_in, "frames_cluster": frames_cluster}
        savemat(path2file + '_Full' + '.mat', matdata)
        # np.savez(path2file+'.npz', frames_in, frames_cluster)

    # --- Ending
    print("... This is the end")


if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Settings
    separate_files = True
    path2file = 'data/denoising_dataset'
    # 0: no aligning, 1: maximum, 2: minimum, 3: maximum positive slop, 4: maximum negative slope
    align_mode = 3

    LoadSpAIke_Data(
        path2file=path2file,
        use_fulldata=False,
        separate_files=separate_files,
        align_mode=align_mode
    )

