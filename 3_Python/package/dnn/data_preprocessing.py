import numpy as np
import torch
from tqdm import tqdm
from package.metric import calculate_snr
from package.data.process_noise import frame_noise
from package.data.data_call_cellbib import CellSelector


def change_frame_size(frames_in: np.ndarray, sel_pos: list) -> np.ndarray:
    """Reducing the frame size of input frames to specific values

    Args:
        frames_in: input_values
    """
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


def calculate_frame_mean(
        frames_in: np.ndarray,
        frames_cl: np.ndarray
    ) -> np.ndarray:
    """Calculating mean waveforms of spike waveforms"""
    NoCluster, NumCluster = np.unique(frames_cl, return_counts=True)
    SizeCluster = np.size(NoCluster)

    frames_out = np.zeros(shape=(SizeCluster, frames_in.shape[1]))
    for idx0, val in enumerate(NoCluster):
        # --- Mean waveform
        indices = np.argwhere(frames_cl == val).flatten()
        frames_out[idx0, :] = np.mean(frames_in[indices, :], axis=0)

    return frames_out.astype(int)


def calculate_frame_median(
        frames_in: np.ndarray,
        frames_cl: np.ndarray
    ) -> np.ndarray:
    """Calculating mean waveforms of spike waveforms with median()"""
    NoCluster, NumCluster = np.unique(frames_cl, return_counts=True)
    SizeCluster = np.size(NoCluster)

    frames_out = np.zeros(shape=(SizeCluster, frames_in.shape[1]))
    for idx0, val in enumerate(NoCluster):
        # --- Mean waveform
        indices = np.argwhere(frames_cl == val).flatten()
        frames_out[idx0, :] = np.median(frames_in[indices, :], axis=0)

    return frames_out.astype(int)


def calculate_frame_snr(
        frames_in: np.ndarray,
        frames_cl: np.ndarray,
        frames_mean: np.ndarray
) -> np.ndarray:
    """Calculating SNR of each cluster"""""
    NoCluster, NumCluster = np.unique(frames_cl, return_counts=True)

    cluster_snr = np.zeros(shape=(NumCluster.size, 4), dtype=float)
    for idx, id in enumerate(NoCluster):
        indices = np.where(frames_cl == id)[0]
        snr0 = np.zeros(shape=(indices.size,), dtype=float)

        for i, frame in enumerate(frames_in[indices, :]):
            snr0[i] = calculate_snr(frame, frames_mean[id, :])

        cluster_snr[idx, 0] = np.min(snr0)
        cluster_snr[idx, 1] = np.mean(snr0)
        cluster_snr[idx, 2] = np.max(snr0)
        cluster_snr[idx, 3] = i

    return cluster_snr


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


class DataNormalization:
    """Normalizing the input data to enhance classification performance.

    Args:
        mode (str): The processing mode, can be one of "CPU", "GPU", or "FPGA".
        method (str): The normalization method, can be one of "minmax", "binary", "norm", "zscore", "medianmad", or "meanmad".
        do_bipolar (bool): Boolean indicating whether to use bipolar normalization.
        do_global (bool): Boolean indicating whether to use global normalization.

    Methods:
        normalize(): Normalize the input data based on the selected mode and method.

    Examples:
        # Create an instance of DataNormalization
        data_normalizer = DataNormalization(frames_in, mode="GPU", method="minmax", do_bipolar=True, do_global=False)

        # Normalize the data
        normalized_frames = data_normalizer.normalize(frames_in: np.ndarray)
            frames_in: Input data to be normalized.
    """
    def __init__(self, mode, method, do_bipolar, do_global):
        self.mode = mode
        self.method = method
        self.do_bipolar = do_bipolar
        self.do_global = do_global

    def _normalize_cpu(self, frames_in: np.ndarray):
        mean_val = 0 if self.do_bipolar else 0.5
        scale_mean = 1 if self.do_bipolar else 2
        scale_global = np.max([np.max(frames_in), -np.min(frames_in)]) if self.do_global else 1

        std_global = np.std(frames_in) if self.do_global else 1
        mean_global = np.mean(frames_in) if self.do_global else 1
        median_global = np.median(frames_in) if self.do_global else 1
        mad_global = np.median(np.absolute(frames_in - np.median(frames_in))) if self.do_global else 1

        frames_out = np.zeros(shape=frames_in.shape)

        match self.method:
            case "minmax":
                for i, frame in enumerate(frames_in):
                    scale_local = np.max([np.max(frame), -np.min(frame)]) if not self.do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    frames_out[i, :] = mean_val + frame / scale

            case "binary":
                for i, frame in enumerate(frames_in):
                    scale_local = np.max([np.max(frame), -np.min(frame)]) if not self.do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    division_value = 0
                    while scale > (2 ** division_value):
                        division_value += 1
                    frames_out[i, :] = mean_val + frame / (2 ** division_value)

            case "norm":
                for i, frame in enumerate(frames_in):
                    scale = np.linalg.norm(frame)
                    frames_out[i, :] = frame / scale

            case "zscore":
                for i, frame in enumerate(frames_in):
                    std_local = np.std(frame) if not self.do_global else 1
                    mean_local = np.mean(frame) if not self.do_global else 1
                    mean = mean_local * mean_global
                    std = std_local * std_global
                    frames_out[i, :] = (frame - mean) / std

            case "medianmad":
                for i, frame in enumerate(frames_in):
                    median_local = np.median(frame) if not self.do_global else 1
                    mad_local = np.median(np.absolute(frame - np.median(frame))) if not self.do_global else 1
                    median = median_local * median_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - median) / mad

            case "meanmad":
                for i, frame in enumerate(frames_in):
                    mean_local = np.mean(frame) if not self.do_global else 1
                    mad_local = np.mean(np.absolute(frame - np.mean(frame))) if not self.do_global else 1
                    mean = mean_local * mean_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - mean) / mad

        return frames_out

    def _normalize_gpu(self, frames_in: torch.Tensor):
        mean_val = 0 if self.do_bipolar else 0.5
        scale_mean = 1 if self.do_bipolar else 2
        scale_global = torch.max(torch.abs(frames_in)).item() if self.do_global else 1

        std_global = torch.std(frames_in).item() if self.do_global else 1
        mean_global = torch.mean(frames_in).item() if self.do_global else 1
        median_global = torch.median(frames_in).item() if self.do_global else 1
        mad_global = torch.median(
            torch.abs(frames_in - torch.median(frames_in))).item() if self.do_global else 1

        frames_out = torch.zeros_like(frames_in)

        match self.method:
            case "minmax":
                for i, frame in enumerate(frames_in):
                    scale_local = torch.max(torch.abs(frame)).item() if not self.do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    frames_out[i, :] = mean_val + frame / scale

            case "binary":
                for i, frame in enumerate(frames_in):
                    scale_local = torch.max(torch.abs(frame)).item() if not self.do_global else 1
                    scale = scale_mean * scale_local * scale_global
                    division_value = 0
                    while scale > (2 ** division_value):
                        division_value += 1
                    frames_out[i, :] = mean_val + frame / (2 ** division_value)

            case "norm":
                for i, frame in enumerate(frames_in):
                    scale = torch.norm(frame)
                    frames_out[i, :] = frame / scale

            case "zscore":
                for i, frame in enumerate(frames_in):
                    std_local = torch.std(frame) if not self.do_global else 1
                    mean_local = torch.mean(frame) if not self.do_global else 1
                    mean = mean_local * mean_global
                    std = std_local * std_global
                    frames_out[i, :] = (frame - mean) / std

            case "medianmad":
                for i, frame in enumerate(frames_in):
                    median_local = torch.median(frame) if not self.do_global else 1
                    mad_local = torch.median(torch.abs(frame - torch.median(frame))) if not self.do_global else 1
                    median = median_local * median_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - median) / mad

            case "meanmad":
                for i, frame in enumerate(frames_in):
                    mean_local = torch.mean(frame) if not self.do_global else 1
                    mad_local = torch.mean(torch.abs(frame - torch.mean(frame))) if not self.do_global else 1
                    mean = mean_local * mean_global
                    mad = mad_local * mad_global
                    frames_out[i, :] = (frame - mean) / mad

        return frames_out

    def _normalize_fpga(self, frames_in: np.ndarray, simple_method=False) -> np.ndarray:
        mean_val = 0 if self.do_bipolar else 0.5
        scale_mean = 1 if self.do_bipolar else 2
        scale_global = np.max([np.max(frames_in), -np.min(frames_in)]) if self.do_global else 1

        frames_out = np.zeros(shape=frames_in.shape)

        for i, frame in enumerate(frames_in):
            scale_local = np.max([np.max(frame), -np.min(frame)]) if not self.do_global else 1
            scale = scale_mean * scale_local * scale_global
            division_value = 1

            while scale > (2 ** division_value):
                division_value += 1

            maximum = scale_global if self.do_global else scale_local
            adjust_maximum = maximum
            divider = 2**division_value if self.do_bipolar else 2 ** (division_value - 1)
            coeff = [0, 0, 0, 0]
            for j in range(1, 5):
                if adjust_maximum + adjust_maximum / (2 ** j) <= divider:
                    adjust_maximum = adjust_maximum + adjust_maximum / (2**j)
                    coeff[j - 1] = 1
            if simple_method:
                frames_out[i, :] = mean_val + frame / (2 ** division_value)
            else:
                frames_out[i, :] = mean_val + (frame + coeff[0]*frame/2**1 + coeff[1]*frame/2**2 + coeff[2]*frame/2**3 + coeff[3]*frame/2**4) / (2 ** division_value)
        return frames_out

    def normalize(self, frames_in: np.ndarray):
        match self.mode:
            case "CPU":
                frames_out = self._normalize_cpu(frames_in)
            case "GPU":
                frames_in = torch.from_numpy(frames_in)
                frames_out = self._normalize_gpu(frames_in)
            case "FPGA":
                frames_out = self._normalize_fpga(frames_in)
            case _:
                print("Selected mode is not available.")
                return 0

        return frames_out
