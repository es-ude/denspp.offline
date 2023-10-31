import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# TODO: Add normal training of denoising autoencoder
class DatasetAE(Dataset):
    """Dataset Preparator for training Autoencoder"""
    def __init__(self, frames: np.ndarray, index: np.ndarray,
                 mean_frame: np.ndarray,
                 mode_train=0):
        self.frames_orig = np.array(frames, dtype=np.float32)
        self.frames_noise = np.array(frames, dtype=np.float32)
        self.frames_mean = np.array(mean_frame, dtype=np.float32)
        self.cluster = index

        self.mode_train = mode_train
        if mode_train == 1:
            self.data_type = "Denoising Autoencoder (mean)"
        elif mode_train == 2:
            self.data_type = "Denoising Autoencoder (Add noise)"
        else:
            self.data_type = "Autoencoder"

    def __len__(self):
        return self.cluster.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.cluster[idx]
        frame_mean = self.frames_mean[cluster_id, :]

        if self.mode_train == 1:
            # Denoising Autoencoder Training with mean
            frame_in = self.frames_orig[idx, :]
            frame_out = self.frames_mean[cluster_id, :]
        elif self.mode_train == 2:
            # Denoising Autoencoder Training with adding noise on input
            frame_in = self.frames_noise[idx, :]
            frame_out = self.frames_orig[idx, :]
        else:
            # Normal Autoencoder Training
            frame_in = self.frames_orig[idx, :]
            frame_out = self.frames_orig[idx, :]

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
