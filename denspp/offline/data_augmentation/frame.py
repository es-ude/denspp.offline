import numpy as np
from denspp.offline.analog import SettingsNoise, ProcessNoise
from denspp.offline.metric import calculate_snr


def augmentation_mean_waveform(
        frames_mean: np.ndarray,
        frames_cl: np.ndarray,
        snr_in: np.ndarray,
        num_min_frames: int,
        fs: float=20e3
) -> dict:
    """Tool for data augmentation of input spike frames (with mean waveform)
    :param frames_mean:         Numpy array with mean waveform
    :param frames_cl:           Numpy array with corresponding cluster id to each waveform
    :param snr_in:              Signal-to-Noise ratio (SNR) in dB of each waveform
    :param num_min_frames:      Minimum number of frames to augment
    :param fs:                  Sampling frequency for noise generation [Hz]
    :return:                    Dict with (1) numpy array of augmented frames and (2) corresponding ids
    """
    frames_out = np.array([], dtype=frames_mean.dtype)
    cluster_out = np.array([], dtype=frames_cl.dtype)

    # --- Adding artificial noise frames (Augmented Path)
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)
    for idx0, (id, num) in enumerate(zip(id_cluster, num_cluster)):
        snr_range = [snr_in[idx0, 0], snr_in[idx0, 2]]
        num_frames = num_min_frames - num
        new_frame = _frame_noise(
            num_frames=num_frames,
            frame_in=frames_mean[idx0, :],
            snr_out=snr_range,
            fs=fs
        )[1]
        new_cluster = np.zeros(shape=(num_frames,)) + id
        frames_out = new_frame if idx0 == 0 else np.append(frames_out, new_frame, axis=0)
        cluster_out = new_cluster if idx0 == 0 else np.append(cluster_out, new_cluster, axis=0)
    return {'frames': frames_out, 'id': cluster_out}


def augmentation_changing_position(
        frames_in: np.ndarray,
        frames_cl: np.ndarray,
        num_min_frames: int
) -> dict:
    """Tool for data augmentation of input spike frames using switching positions (change position)
    :param frames_in:           Numpy array with mean waveform
    :param frames_cl:           Numpy array with corresponding cluster id to each waveform
    :param num_min_frames:      Minimum number of frames to augment
    :return:                    Dict with (1) numpy array of augmented frames and (2) corresponding IDs
    """
    out_frames = np.array([], dtype=frames_in.dtype)
    out_cluster = np.array([], dtype=frames_cl.dtype)
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)
    # --- Adding artificial noise frames (Augmented Path)
    max_y = np.max(num_cluster)
    max_x = frames_in.shape[1]

    for idx, (id, val) in enumerate(zip(id_cluster, num_cluster)):
        xpos_frames = np.where(frames_cl == id)[0]
        sel_frames = frames_in[xpos_frames, :]

        no_frames = num_min_frames + max_y - val
        new_frame = np.zeros(shape=(no_frames, frames_in.shape[1]), dtype=frames_in.dtype)
        new_cluster = np.zeros(shape=(no_frames, ), dtype=frames_cl.dtype) + id_cluster[idx]
        sel_position = np.random.randint(low=0, high=sel_frames.shape[0], size=(no_frames, max_x))
        # --- Generating frames
        for idx0, pos0 in enumerate(sel_position):
            for idx1, pos1 in enumerate(pos0):
                new_frame[idx0, idx1] = sel_frames[pos1, idx1]

        out_frames = new_frame if idx == 0 else np.append(out_frames, new_frame, axis=0)
        out_cluster = new_cluster if idx == 0 else np.append(out_cluster, new_cluster, axis=0)
    return {
        'frames': np.append(frames_in, out_frames, axis=0),
        'id': np.append(frames_cl, out_cluster, axis=0)
    }


def augmentation_reducing_samples(
        frames_in: np.ndarray,
        frames_cl: np.ndarray,
        num_frames: int,
        do_shuffle: bool=True
) -> dict:
    """Tool for data augmentation of input spike frames (change position)
    :param frames_in:           Numpy array with mean waveform
    :param frames_cl:           Numpy array with corresponding cluster id to each waveform
    :param num_frames:          Minimum number of frames to augment
    :param do_shuffle:          Whether to shuffle samples
    :return:                    Dict with (1) numpy array of augmented frames and (2) IDs
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


def _frame_noise(num_frames: int, frame_in: np.ndarray, snr_out: list, fs: float, return_integer: bool=False) -> tuple[np.ndarray, np.ndarray]:
    settings_noise = SettingsNoise(
        temp=300,
        wgn_dB=-120,
        Fc=10,
        slope=0.6
    )
    handler_noise = ProcessNoise(settings_noise, fs)

    width = frame_in.size
    frames_noise = np.zeros(shape=(num_frames, width))
    frames_out = np.zeros(shape=(num_frames, width))
    for idx in range(0, num_frames):
        SNR_soll = np.random.uniform(snr_out[0], snr_out[1])
        SNR_diff = 100
        noise_lvl = -80

        spk_random = np.array(4 * (np.random.rand(frame_in.size) -0.5), dtype=int) if return_integer else 1e-9 * (np.random.rand(frame_in.size) -0.5)
        spk = frame_in + spk_random
        noise = np.zeros(shape=(width, ))
        while np.abs(SNR_diff) > 0.02:
            noise = handler_noise.gen_noise_real_pwr(width, noise_lvl)
            SNR_ist = calculate_snr(spk + noise, spk)
            SNR_diff = SNR_ist - SNR_soll
            noise_lvl += SNR_diff / 10

        frames_out[idx, :] = np.array(frame_in + noise, dtype=int) if return_integer else frame_in + noise
        frames_noise[idx, :] = np.array(noise, dtype=int) if return_integer else noise
    return frames_noise, frames_out


def generate_zero_frames(frame_size: int, num_frames: int,
                         snr_range: list, fs: float=20e3, return_int: bool=False
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generating zero frames with noise for data augmentation
    :param frame_size:  Integer with frame size (number of samples for each frame)
    :param num_frames:  Integer with number of frames to generate
    :param snr_range:   List with two elemente to define the range to generate noise
    :param fs:          Sampling rate  [Hz]
    :param return_int:  Return frames as integer (True) or float (False)
    :return:            Tuple with [0] new zero frames, [1] new cluster ids and [2] mean frame
    """
    frames = _frame_noise(
        num_frames=num_frames,
        frame_in=np.zeros(shape=(frame_size, )),
        snr_out=snr_range,
        fs=fs,
        return_integer=return_int
    )[1]
    cluster = np.zeros(shape=(frames.shape[0], ), dtype=int)
    return frames, cluster, np.mean(frames, axis=0)


def calculate_frame_mean(frames_in: np.ndarray,
                         frames_cl: np.ndarray,
                         return_int: bool=False
                         ) -> np.ndarray:
    """Calculating mean waveforms of spike waveforms
    :param frames_in:   Numpy array with raw waveforms
    :param frames_cl:   Numpy array with cluster IDs
    :param return_int:  Boolean flag for returning result as integer otherwise float
    :return:            Numpy array with mean waveforms for each class
    """
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)
    size_cluster = np.size(id_cluster)

    frames_out = np.zeros(shape=(size_cluster, frames_in.shape[1]))
    for idx0, val in enumerate(id_cluster):
        # --- Mean waveform
        indices = np.argwhere(frames_cl == val).flatten()
        frames_out[idx0, :] = np.mean(frames_in[indices, :], axis=0)

    return frames_out.astype(int) if return_int else frames_out
