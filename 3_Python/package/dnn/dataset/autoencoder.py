import numpy as np
from datetime import datetime
from scipy.io import loadmat
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader

from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.data_preprocessing import change_frame_size, calculate_frame_mean, calculate_frame_snr, generate_zero_frames, data_normalization
from package.dnn.data_augmentation import *


# TODO: Add normal training of denoising autoencoder
class DatasetAE(Dataset):
    """Dataset Preparator for training Autoencoder"""
    def __init__(self, frames: np.ndarray, index: np.ndarray,
                 mean_frame: np.ndarray,
                 mode_train=0):
        self.__frames_orig = np.array(frames, dtype=np.float32)
        self.__frames_noise = np.array(frames, dtype=np.float32)
        self.__frames_mean = np.array(mean_frame, dtype=np.float32)
        self.__cluster = index

        self.mode_train = mode_train
        if mode_train == 1:
            self.data_type = "Denoising Autoencoder (mean)"
        elif mode_train == 2:
            self.data_type = "Denoising Autoencoder (Add noise)"
        else:
            self.data_type = "Autoencoder"

    def __len__(self):
        return self.__cluster.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.__cluster[idx]
        frame_mean = self.__frames_mean[cluster_id, :]

        if self.mode_train == 1:
            # Denoising Autoencoder Training with mean
            frame_in = self.__frames_orig[idx, :]
            frame_out = self.__frames_mean[cluster_id, :]
        elif self.mode_train == 2:
            # Denoising Autoencoder Training with adding noise on input
            frame_in = self.__frames_noise[idx, :]
            frame_out = self.__frames_orig[idx, :]
        else:
            # Normal Autoencoder Training
            frame_in = self.__frames_orig[idx, :]
            frame_out = self.__frames_orig[idx, :]

        return {'in': frame_in, 'out': frame_out, 'cluster': cluster_id, 'mean': frame_mean}


def prepare_plotting(data_plot: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Getting data from DataLoader for Plotting Results"""
    din = []
    dout = []
    did = []
    dmean = []
    for idx, vdata in enumerate(data_plot):
        din = vdata['in'] if idx == 0 else np.append(din, vdata['in'], axis=0)
        dout = vdata['out'] if idx == 0 else np.append(dout, vdata['out'], axis=0)
        dmean = vdata['mean'] if idx == 0 else np.append(dmean, vdata['mean'], axis=0)
        did = vdata['cluster'] if idx == 0 else np.append(did, vdata['cluster'])

    return din, dout, did, dmean


def prepare_training(path: str, settings: Config_PyTorch, mode_train_ae=0) -> DatasetAE:
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
        frames_cl = npzfile['arr_2']
    else:
        # --- MATLAB reading file
        npzfile = loadmat(path)
        frames_in = npzfile["frames_in"]
        frames_cl = npzfile["frames_cluster"].flatten()
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    # --- Mean waveform calculation and data augmentation
    frames_in = change_frame_size(frames_in, settings.data_sel_pos)
    frames_mean = calculate_frame_mean(frames_in, frames_cl)
    # plt_memristor_ref(frames_in, frames_cl, frames_mean)

    # --- PART: Exclusion of selected clusters
    if len(settings.data_exclude_cluster) == 0:
        frames_in = frames_in
        frames_cl = frames_cl
    else:
        for i, id in enumerate(settings.data_exclude_cluster):
            selX = np.where(frames_cl != id)
            frames_in = frames_in[selX[0], :]
            frames_cl = frames_cl[selX]

    # --- PART: Reducing samples per cluster (if too large)
    if settings.data_do_reduce_samples_per_cluster:
        print("... do data augmentation with reducing the samples per cluster")
        frames_in, frames_cl = augmentation_reducing_samples(frames_in, frames_cl,
                                                             settings.data_num_samples_per_cluster,
                                                             settings.data_do_shuffle)

    # --- PART: Calculate SNR if desired
    if settings.data_do_augmentation or settings.data_do_addnoise_cluster:
        snr_mean = calculate_frame_snr(frames_in, frames_cl, frames_mean)
    else:
        snr_mean = np.zeros(0, dtype=float)

    # --- PART: Data Augmentation
    if settings.data_do_augmentation and not settings.data_do_reduce_samples_per_cluster:
        print("... do data augmentation")
        # new_frames, new_clusters = augmentation_mean_waveform(
        # frames_mean, frames_cl, snr_mean, settings.data_num_augmentation)
        new_frames, new_clusters = augmentation_change_position(
            frames_in, frames_cl, snr_mean, settings.data_num_augmentation)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cl = np.append(frames_cl, new_clusters, axis=0)

    # --- PART: Generate and add noise cluster
    if settings.data_do_addnoise_cluster:
        snr_range_zero = [np.mean(snr_mean[:, 0]), np.mean(snr_mean[:, 2])]
        info = np.unique(frames_cl, return_counts=True)
        num_cluster = np.max(info[0]) + 1
        num_frames = np.max(info[1])
        print(f"... adding a zero-noise cluster: cluster = {num_cluster} - number of frames = {num_frames}")

        new_mean, new_clusters, new_frames = generate_zero_frames(frames_in.shape[1], num_frames, snr_range_zero)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cl = np.append(frames_cl, num_cluster + new_clusters, axis=0)
        frames_mean = np.vstack([frames_mean, new_mean])

    # --- PART: Data Normalization
    if settings.data_do_normalization:
        frames_in = data_normalization(frames_in)
        frames_mean = data_normalization(frames_mean)

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    print(f"... used data points for training: class = {check[0]} and num = {check[1]}")
    return DatasetAE(frames_in, frames_cl, frames_mean, mode_train_ae)
