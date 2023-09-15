import numpy as np
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