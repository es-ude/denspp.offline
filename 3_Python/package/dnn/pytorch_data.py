import numpy as np
from datetime import datetime
from scipy.io import loadmat

from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.data_preprocessing import change_frame_size, calculate_mean_waveform, generate_zero_frames, data_normalization
from package.dnn.data_augmentation import augmentation_change_position
from package.dnn.dataset.autoencoder import DatasetAE
from package.dnn.dataset.spike_detection import DatasetSDA


def prepare_training_spike_frame(path: str, settings: Config_PyTorch, mode_train_ae=0) -> DatasetAE:
    """Preparing datasets incl. augmentation for spike-frame based training (without pre-processing)"""

    # --- Pre-definitions
    str_datum = datetime.now().strftime('%Y%m%d %H%M%S')
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    # --- Data loading
    if path[-3:] == "npz":
        # --- NPZ reading file
        npzfile = np.load(path)
        frames_in = npzfile['arr_0']
        frames_cluster = npzfile['arr_2']
    else:
        # --- MATLAB reading file
        npzfile = loadmat(path)
        frames_in = npzfile["frames_in"]
        frames_cluster = npzfile["frames_cluster"].flatten()
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    # --- Mean waveform calculation and data augmentation
    frames_in = change_frame_size(frames_in, settings.data_sel_pos)
    frames_mean, snr_mean = calculate_mean_waveform(frames_in, frames_cluster)
    # plt_memristor_ref(frames_in, frames_cluster, frames_mean)

    # --- PART: Exclusion of selected clusters
    if len(settings.data_exclude_cluster) == 0:
        frames_in = frames_in
        frames_cluster = frames_cluster
    else:
        for i, id in enumerate(settings.data_exclude_cluster):
            selX = np.where(frames_cluster != id)
            frames_in = frames_in[selX[0], :]
            frames_cluster = frames_cluster[selX]

    # --- PART: Data Augmentation
    if settings.data_do_augmentation:
        print("... do data augmentation")
        # new_frames, new_clusters = augmentation_mean_waveform(
        # frames_mean, frames_cluster, snr_mean, settings.data_num_augmentation)
        new_frames, new_clusters = augmentation_change_position(
            frames_in, frames_cluster, snr_mean, settings.data_num_augmentation)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cluster = np.append(frames_cluster, new_clusters, axis=0)

    # --- PART: Generate and add noise cluster
    if settings.data_do_addnoise_cluster:
        snr_range_zero = [np.mean(snr_mean[:, 0]), np.mean(snr_mean[:, 2])]
        info = np.unique(frames_cluster, return_counts=True)
        num_cluster = np.max(info[0]) + 1
        num_frames = np.max(info[1])
        print(f"... adding a zero-noise cluster: cluster = {num_cluster} - number of frames = {num_frames}")

        new_mean, new_clusters, new_frames = generate_zero_frames(frames_in.shape[1], num_frames, snr_range_zero)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cluster = np.append(frames_cluster, num_cluster + new_clusters, axis=0)
        frames_mean = np.vstack([frames_mean, new_mean])

    # --- PART: Data Normalization
    if settings.data_do_normalization:
        frames_in = data_normalization(frames_in)
        frames_mean = data_normalization(frames_mean)

    # --- Output
    check = np.unique(frames_cluster, return_counts=True)
    print(f"... used data points for training: class = {check[0]} and num = {check[1]}")
    return DatasetAE(frames_in, frames_cluster, frames_mean, mode_train_ae)


def prepare_training_sda(path: str, settings: Config_PyTorch) -> DatasetSDA:
    """Preparing datasets incl. augmentation for spike-detection-based training (without pre-processing)"""
    # --- Pre-definitions
    str_datum = datetime.now().strftime('%Y%m%d %H%M%S')
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    # --- MATLAB reading file
    npzfile = loadmat(path)
    frames_in = npzfile["sda_in"]
    frames_pred = npzfile["sda_pred"]
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    return DatasetSDA(frames_in, frames_pred, 3)
