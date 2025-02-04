import numpy as np
#from .frame_preprocessing import generate_frames


def augmentation_mean_waveform(
        frames_mean: np.ndarray,
        frames_cl: np.ndarray,
        snr_in: np.ndarray,
        num_min_frames: int) -> dict:
    """Tool for data augmentation of input spike frames (with mean waveform)
    :param frames_mean:         Numpy array with mean waveform
    :param frames_cl:           Numpy array with corresponding cluster id to each waveform
    :param snr_in:              Signal-to-Noise ratio (SNR) in dB of each waveform
    :param num_min_frames:      Minimum number of frames to augment
    :return:                    Dict with (1) numpy array of augmented frames and (2)
    """

    frames_out = np.array([], dtype=frames_mean.dtype)
    cluster_out = np.array([], dtype=frames_cl.dtype)
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)
    # --- Adding artificial noise frames (Augmented Path)
    max_y = np.max(num_cluster)

    for idx0, val in enumerate(num_cluster):
        snr_range = [snr_in[idx0, 0], snr_in[idx0, 2]]
        no_frames = num_min_frames + max_y - val
        (new_cluster, new_frame) = generate_frames(no_frames, frames_mean[idx0, :], id_cluster[idx0], snr_range)
        # Adding to output
        frames_out = new_frame if idx0 == 0 else np.append(frames_out, new_frame, axis=0)
        cluster_out = new_cluster if idx0 == 0 else np.append(cluster_out, new_cluster, axis=0)

    return {'frames': frames_out, 'id': cluster_out}


def augmentation_change_position(
        frames_in: np.ndarray,
        frames_cl: np.ndarray,
        num_min_frames: int) -> dict:
    """Tool for data augmentation of input spike frames (change position)
    :param frames_in:           Numpy array with mean waveform
    :param frames_cl:           Numpy array with corresponding cluster id to each waveform
    :param num_min_frames:      Minimum number of frames to augment
    :return:                    Dict with (1) numpy array of augmented frames and (2)
    """

    out_frames = np.array([], dtype=frames_in.dtype)
    out_cluster = np.array([], dtype=frames_cl.dtype)
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)
    # --- Adding artificial noise frames (Augmented Path)
    max_y = np.max(num_cluster)
    max_x = frames_in.shape[1]

    for idx, val in enumerate(num_cluster):
        xpos_frames = np.where(frames_cl == id_cluster[idx])[0]
        sel_frames = frames_in[xpos_frames, :]

        no_frames = num_min_frames + max_y - val
        new_frame = np.zeros(shape=(no_frames, frames_in.shape[1]), dtype=frames_in.dtype)
        new_cluster = np.zeros(shape=(no_frames, ), dtype=frames_cl.dtype) + id_cluster[idx]
        sel_position = np.random.randint(low=0, high=sel_frames.shape[0], size=(no_frames, max_x))
        # --- Generating frames
        for idx0, pos0 in enumerate(sel_position):
            for idx1, pos1 in enumerate(pos0):
                new_frame[idx0, idx1] = sel_frames[pos1, idx1]

        # Adding to output
        out_frames = new_frame if idx == 0 else np.append(out_frames, new_frame, axis=0)
        out_cluster = new_cluster if idx == 0 else np.append(out_cluster, new_cluster, axis=0)

    return {'frames': out_frames, 'id': out_cluster}


def augmentation_reducing_samples(
        frames_in: np.ndarray,
        frames_cl: np.ndarray,
        num_frames: int,
        do_shuffle=True) -> dict:
    """Tool for data augmentation of input spike frames (change position)
    :param frames_in:           Numpy array with mean waveform
    :param frames_cl:           Numpy array with corresponding cluster id to each waveform
    :param num_frames:          Minimum number of frames to augment
    :param do_shuffle:          Whether to shuffle samples
    :return:                    Dict with (1) numpy array of augmented frames and (2)
    """
    cluster_no = np.unique(frames_cl)
    frames_out = np.zeros(1)
    frames_clo = np.zeros(1)

    for ite, id0 in enumerate(cluster_no):
        pos = np.argwhere(frames_cl == id0).flatten()
        if do_shuffle:
            for idx in range(0, 5):
                np.random.shuffle(pos)
        pos = pos[:num_frames]
        frames_out = frames_in[pos, :] if ite == 0 else np.append(frames_out, frames_in[pos, :], axis=0)
        frames_clo = frames_cl[pos] if ite == 0 else np.append(frames_clo, frames_cl[pos], axis=0)

    return {'frames': frames_out, 'id': frames_clo}
