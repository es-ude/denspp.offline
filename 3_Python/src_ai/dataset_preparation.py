import numpy as np
from datetime import datetime
from scipy.io import loadmat

from src_ai.data_preprocessing import change_frame_size, calculate_mean_waveform, generate_zero_frames, data_normalization
from src_ai.data_augmentation import augmentation_mean_waveform, augmentation_change_position


def prepare_training(path: str, excludeCluster: list, sel_pos: list, num_new_frames: int,
                     do_augmentation=False, do_norm=False, do_zeroframes=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Einlesen des Datensatzes inkl. Augmentierung (Kein Pre-Processing)"""
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
    frames_in = change_frame_size(frames_in, sel_pos)
    frames_mean, snr_mean = calculate_mean_waveform(frames_in, frames_cluster)

    # --- PART: Exclusion of selected clusters
    if len(excludeCluster) == 0:
        frames_in = frames_in
        frames_cluster = frames_cluster
    else:
        for i, id in enumerate(excludeCluster):
            selX = np.where(frames_cluster != id)
            frames_in = frames_in[selX[0], :]
            frames_cluster = frames_cluster[selX]

    # --- PART: Data Augmentation
    if do_augmentation:
        print("... do data augmentation")
        # new_frames, new_clusters = augmentation_mean_waveform(frames_mean, frames_cluster, snr_mean, num_new_frames)
        new_frames, new_clusters = augmentation_change_position(frames_in, frames_cluster, snr_mean, num_new_frames)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cluster = np.append(frames_cluster, new_clusters, axis=0)

    # --- PART: Generate and add noise cluster
    if do_zeroframes:
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
    if do_norm:
        frames_in = data_normalization(frames_in)
        frames_mean = data_normalization(frames_mean)

    # --- Output
    check = np.unique(frames_cluster, return_counts=True)
    print(f"... used data points for training: class = {check[0]} and num = {check[1]}")
    return frames_in, frames_cluster, frames_mean
