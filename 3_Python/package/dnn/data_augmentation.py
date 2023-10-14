import numpy as np
from package.dnn.data_preprocessing import generate_frames

def augmentation_mean_waveform(
        frames_mean: np.ndarray,
        frames_cluster: np.ndarray,
        snr_in: np.ndarray,
        num_min_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
    """Tool for data augmentation of input spike frames (with mean waveform)"""

    frames_out = np.array([], dtype=frames_mean.dtype)
    cluster_out = np.array([], dtype=frames_cluster.dtype)
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


def augmentation_change_position(
        frames_in: np.ndarray,
        frames_cluster: np.ndarray,
        snr_in: np.ndarray,
        num_min_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
    """Tool for data augmentation of input spike frames (change position)"""

    out_frames = np.array([], dtype=frames_in.dtype)
    out_cluster = np.array([], dtype=frames_cluster.dtype)
    NoCluster, NumCluster = np.unique(frames_cluster, return_counts=True)
    # --- Adding artificial noise frames (Augmented Path)
    maxY = np.max(NumCluster)
    maxX = frames_in.shape[1]

    for idx, val in enumerate(NumCluster):
        xpos_frames = np.where(frames_cluster == NoCluster[idx])[0]
        sel_frames = frames_in[xpos_frames, :]

        no_frames = num_min_frames + maxY - val
        new_frame = np.zeros(shape=(no_frames, frames_in.shape[1]), dtype=frames_in.dtype)
        new_cluster = np.zeros(shape=(no_frames, ), dtype=frames_cluster.dtype) + NoCluster[idx]
        sel_position = np.random.randint(low=0, high=sel_frames.shape[0], size=(no_frames, maxX))
        # --- Generating frames
        for idx0, pos0 in enumerate(sel_position):
            for idx1, pos1 in enumerate(pos0):
                new_frame[idx0, idx1] = sel_frames[pos1, idx1]

        # Adding to output
        out_frames = new_frame if idx == 0 else np.append(out_frames, new_frame, axis=0)
        out_cluster = new_cluster if idx == 0 else np.append(out_cluster, new_cluster, axis=0)

    return out_frames, out_cluster