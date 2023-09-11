import numpy as np
from datetime import datetime
from scipy.io import loadmat

from src.metric import calculate_snr
from src.processing_noise import frame_noise


def change_frame_size(frames_in: np.ndarray, sel_pos: list) -> np.ndarray:
    """Reducing the frame size of input frames to specific values"""
    if len(sel_pos) != 2:
        # Alle Werte übernehmen
        frames_out = frames_in
    else:
        # Fensterung der Frames
        frames_out = frames_in[:, sel_pos[0]:sel_pos[1]]

    return frames_out


def generate_frames(num: int, frame_in: np.ndarray, cluster_in: int, snr_out: list, fs=20e3) -> [np.ndarray, np.ndarray]:
    """Generating noisy spike frames"""
    new_cluster = cluster_in * np.ones(shape=(num,), dtype=int)
    _, new_frame = frame_noise(num, frame_in, snr_out, fs)

    return new_cluster, new_frame


def generate_zero_frames(frame_size: int, num_frames: int, noise_range: list) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Generating zero frames with noise for data augmentation"""
    mean = 2 + 4 * np.random.randn(1, frame_size)
    out = np.zeros(shape=(frame_size, ), dtype="double")
    (cluster, frames) = generate_frames(num_frames, mean, 0, noise_range)

    return out, cluster, np.round(frames-mean)


def calculate_mean_waveform(
        frames_in: np.ndarray,
        frames_cluster: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Calculating mean waveforms of spike waveforms"""
    NoCluster, NumCluster = np.unique(frames_cluster, return_counts=True)
    SizeCluster = np.size(NoCluster)
    SizeFrame = frames_in.shape[1]

    frames_mean = np.zeros(shape=(SizeCluster, SizeFrame), dtype=int)
    cluster_snr = np.zeros(shape=(SizeCluster, 4), dtype=int)
    for idx0, val in enumerate(NoCluster):
        # --- Mean waveform
        indices = np.where(frames_cluster == val)
        frames_sel = frames_in[indices[0], :]
        mean = np.mean(frames_sel, axis=0, dtype=int)
        frames_mean[idx0, :] = mean

        # --- Calculating SNR
        snr0 = np.zeros(shape=(indices[0].size,), dtype=float)
        for i, frame in enumerate(frames_sel):
            snr0[i] = calculate_snr(frame, mean)

        cluster_snr[idx0, 0] = np.min(snr0)
        cluster_snr[idx0, 1] = np.mean(snr0)
        cluster_snr[idx0, 2] = np.max(snr0)
        cluster_snr[idx0, 3] = i

    return frames_mean, cluster_snr


def data_normalization(
        frames_in: np.ndarray,
        do_bipolar=True,
        do_globalmax=False
    ) -> np.ndarray:
    """Data Normalization of input with range setting do_bipolar (False: [0, 1] - True: [-1, +1])"""
    mean_val = 0 if do_bipolar else 0.5
    scale_mean = 1 if do_bipolar else 2
    scale_global = np.max([np.max(frames_in), -np.min(frames_in)]) if do_globalmax else 1

    frames_out = np.zeros(shape=frames_in.shape)
    for i, frame in enumerate(frames_in):
        scale_local = np.max([np.max(frame), -np.min(frame)]) if not do_globalmax else 1
        scale = scale_mean * scale_local * scale_global
        frames_out[i, :] = mean_val + frame / scale

    return frames_out


def augmentation_data(
        frames_mean: np.ndarray,
        frames_cluster: np.ndarray,
        snr_in: np.ndarray,
        num_min_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
    """Tool for data augmentation of input spike frames"""
    frames_out = np.array([], dtype='float')
    cluster_out = np.array([], dtype='int')

    NoCluster, NumCluster = np.unique(frames_cluster, return_counts=True)

    # --- Adding artificial noise frames (Augmented Path)
    maxY = np.max(NumCluster)

    for idx0, val in enumerate(NumCluster):
        snr_range = [snr_in[idx0, 0], snr_in[idx0, 2]]
        no_frames = num_min_frames + maxY - val
        (new_cluster, new_frame) = generate_frames(no_frames, frames_mean[idx0, :], NoCluster[idx0], snr_range)
        # Adding to output
        frames_out = new_frame if idx0 == 0 else np.append(frames_out, new_frame, axis=0)
        cluster_out = new_cluster if idx0 == 0 else np.append(cluster_out, new_cluster, axis=0)

    return frames_out, cluster_out


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
        new_frames, new_clusters = augmentation_data(frames_mean, frames_cluster, snr_mean, num_new_frames)
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
