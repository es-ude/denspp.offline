import numpy as np
from denspp.offline.metric import calculate_snr
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise
from denspp.offline.data_call.call_cellbib import CellSelector


def change_frame_size(frames_in: np.ndarray, sel_pos: list) -> np.ndarray:
    """Reducing the frame size of input frames to specific values
    Args:
        frames_in:  input_values
        sel_pos:    List with two elements in order to say position start and end
    Returns:
        frames with reduced size
    """
    return frames_in if len(sel_pos) != 2 else frames_in[:, sel_pos[0]:sel_pos[1]]


def generate_frames(num: int, frame_in: np.ndarray, cluster_in: int, snr_out: list, fs=20e3) -> [np.ndarray, np.ndarray]:
    """Generating noisy spike frames"""
    new_cluster = cluster_in * np.ones(shape=(num,), dtype=int)
    _, new_frame = frame_noise(num, frame_in, snr_out, fs)
    return new_cluster, new_frame


def generate_zero_frames(frame_size: int, num_frames: int, noise_range: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generating zero frames with noise for data augmentation"""
    mean = 2 + 4 * np.random.randn(1, frame_size)
    out = np.zeros(shape=(frame_size, ), dtype="double")
    (cluster, frames) = generate_frames(num_frames, mean, 0, noise_range)
    return out, cluster, np.round(frames - mean)


def calculate_frame_mean(frames_in: np.ndarray, frames_cl: np.ndarray,
                         do_integer_output=False) -> np.ndarray:
    """Calculating mean waveforms of spike waveforms"""
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)
    size_cluster = np.size(id_cluster)

    frames_out = np.zeros(shape=(size_cluster, frames_in.shape[1]))
    for idx0, val in enumerate(id_cluster):
        # --- Mean waveform
        indices = np.argwhere(frames_cl == val).flatten()
        frames_out[idx0, :] = np.mean(frames_in[indices, :], axis=0)

    return frames_out.astype(int) if do_integer_output else frames_out


def calculate_frame_median(frames_in: np.ndarray, frames_cl: np.ndarray,
                           do_integer_output=False) -> np.ndarray:
    """Calculating mean waveforms of spike waveforms with median()"""
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)
    size_cluster = np.size(id_cluster)

    frames_out = np.zeros(shape=(size_cluster, frames_in.shape[1]))
    for idx0, val in enumerate(id_cluster):
        # --- Mean waveform
        indices = np.argwhere(frames_cl == val).flatten()
        frames_out[idx0, :] = np.median(frames_in[indices, :], axis=0)

    return frames_out.astype(int) if do_integer_output else frames_out


def calculate_frame_snr(
        frames_in: np.ndarray,
        frames_cl: np.ndarray,
        frames_mean: np.ndarray) -> np.ndarray:
    """Calculating SNR of each cluster"""""
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)

    cluster_snr = np.zeros(shape=(num_cluster.size, 4), dtype=float)
    for idx, id in enumerate(id_cluster):
        indices = np.where(frames_cl == id)[0]
        snr0 = np.zeros(shape=(indices.size,), dtype=float)

        for i, frame in enumerate(frames_in[indices, :]):
            snr0[i] = calculate_snr(frame, frames_mean[id, :])

        cluster_snr[idx, 0] = np.min(snr0)
        cluster_snr[idx, 1] = np.mean(snr0)
        cluster_snr[idx, 2] = np.max(snr0)
        cluster_snr[idx, 3] = i

    return cluster_snr


def frame_noise(no_frames: int, frame_in: np.ndarray, noise_pwr: list, fs: float) -> [np.ndarray, np.ndarray]:
    """Generation of noisy spike frames with AWGN with noise power [dB] in specific interval"""
    width = frame_in.size
    frames_noise = np.zeros(shape=(no_frames, width), dtype="double")
    frames_out = np.zeros(shape=(no_frames, width), dtype="double")
    snr_chck = np.zeros(shape=(no_frames, ), dtype="double")

    settings_noise = SettingsNoise(
        temp=300,
        wgn_dB=-120,
        Fc=10,
        slope=0.6,
        do_print=False
    )
    handler_noise = ProcessNoise(settings_noise, fs)

    # --- Adding noise
    for idx in range(0, no_frames):
        SNR_soll = np.random.uniform(noise_pwr[0], noise_pwr[1])
        SNR_diff = 100
        noise_lvl = -80

        spk = frame_in
        SNR_ist = -1000
        while np.abs(SNR_diff) > 0.02:
            noise = handler_noise.gen_noise_awgn_dev(width, noise_lvl)
            SNR_ist = calculate_snr(spk + noise, spk)
            SNR_diff = SNR_ist - SNR_soll
            noise_lvl += SNR_diff / 10

        snr_chck[idx] = SNR_ist
        frames_out[idx, :] = np.round(frame_in + noise)
        frames_noise[idx, :] = np.round(noise)
    return frames_noise, frames_out

#TODO: Include CellFormat
def reconfigure_cluster_with_cell_lib(path: str, sel_mode_classes: int,
                                      frames_in: np.ndarray, frames_cl: np.ndarray) -> [np.ndarray, np.ndarray, dict]:
    """Function for reducing the samples for a given cell bib"""
    check_class = ['fzj', 'RGC']
    check_path = path[:-4].split("_")
    # --- Check if one is available
    flag = -1
    for path0 in check_path:
        for idx, j in enumerate(check_class):
            if path0 == j:
                flag = idx
                break

    if not flag == -1:
        cl_sampler = CellSelector(flag, sel_mode_classes)
        cell_dict = cl_sampler.get_celltype_names()
        print(f"... Cluster types before reconfiguration: {np.unique(frames_cl)}")
        cluster = cl_sampler.get_class_to_id(frames_cl)

        # Removing undesired samples
        pos = np.argwhere(cluster != -1).flatten()
        print(f"... Cluster types after reconfiguration: {np.unique(cluster)}")
        cell_frame = frames_in[pos, :]
        cell_cl = cluster[pos]
    else:
        cell_frame = frames_in
        cell_cl = frames_cl
        cell_dict = list()

    return cell_frame, cell_cl, cell_dict
