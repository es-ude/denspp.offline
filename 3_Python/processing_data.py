import numpy as np

from settings import Settings
from src.call_data import call_data
from src.afe import AFE
from scipy.io import savemat

def getting_frames(data: np.ndarray, xpos: int, dxneg: int, dxpos: int, cluster: np.ndarray, cluster_no: int):
    frames_out = np.zeros(shape=(xpos.size, dxneg+dxpos))
    mean_value = np.zeros(shape=(np.unique(cluster).size, 1))
    frames_cluster = np.zeros(shape=(xpos.size, 1))

    if cluster_no == []:
        max_val = 0
    else:
        max_val = 1+np.argmax(np.unique(cluster_no))

    idx = 0
    for pos in xpos:
        frames_out[idx, :] = data[pos-dxneg:pos+dxpos]
        frames_cluster[idx] = max_val + cluster[idx]
        mean_value[cluster[idx]] += 1
        idx += 1

    return frames_out, frames_cluster


def LoadSpAIke_Data(path2file: str, use_fulldata: bool):
    settings = Settings()
    afe = AFE(settings)

    #0: no aligning, 1: maximum, 2: minimum, 3: maximum positive slop, 4: maximum negative slope
    align_mode = 3

    # --- Selection of datasets and points
    MaxDataPoints = np.array([5, 16, 22])
    if use_fulldata:
        TakeDatasets = np.array([1, 2, 3])
        # If value = 0 --> All data points will be loaded - otherwise Maximum number
        NoDataPoints = np.array([0, 0, 0])
    else:
        TakeDatasets = np.array([3])
        NoDataPoints = np.array([0])

    # ------ Loading Data: Preparing Data
    print("... loading the datasets")
    frames_in = []
    frames_cluster = []
    runs = 0

    # ------ Loading Data: Running
    iteration = 0
    for runSet in TakeDatasets:
        runPoint = 1
        if NoDataPoints[iteration] == 0:
            endPoint = MaxDataPoints[runSet-1]

        while (runPoint <= endPoint):
            (neuron, labeling) = call_data(
                path2data=settings.path2data,
                data_type=runSet,
                data_set=runPoint,
                desired_fs=settings.desired_fs,
                t_range=settings.t_range,
                ch_sel=settings.ch_sel,
                plot=False
            )

            # Adapting the input for our application
            u_ana, _ = afe.pre_amp(neuron.data)
            x_adc = afe.adc_nyquist(u_ana, 1)

            x_pos = np.floor(labeling.spike_xpos * settings.sample_rate / neuron.fs).astype("int")

            (frame_raw, frame_cluster) = getting_frames(
                data=x_adc, xpos=x_pos,
                dxneg=5, dxpos=15 + afe.frame_length,
                cluster=labeling.cluster_id,
                cluster_no=frames_cluster
            )
            # --- Aligning all frames
            if(0):
                frame_aligned = frame_raw
            else:
                frame_aligned = afe.frame_aligning(frame_raw, align_mode, 1)

            if runs == 0:
                frames_in = frame_aligned
                frames_cluster = frame_cluster
            else:
                frames_in = np.concatenate((frames_in, frame_aligned), axis=0)
                frames_cluster = np.concatenate((frames_cluster, frame_cluster), axis=0)

            # --- Meaning all input frames

            runs += 1
            runPoint += 1

        iteration += 1

    # --- Saving data
    matdata = {"frames_in": frames_in, "frames_cluster": frames_cluster}
    savemat(path2file+'.mat', matdata)
    np.savez(path2file+'.npz', frames_in, frames_cluster)

    # --- Ending
    print("... This is the end")


if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Settings
    path2file = 'data/denoising_dataset'

    LoadSpAIke_Data(
        path2file=path2file,
        use_fulldata=True
    )

