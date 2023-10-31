import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetClass(Dataset):
    """Dataset Preparator for training Spike Detection Classification with Neural Network"""
    def __init__(self, frame: np.ndarray, sda: np.ndarray, slice: int):
        self.frame_slice = np.array(frame, dtype=float)
        self.sda_class = np.array(sda, dtype=bool)
        self.sda_slice = np.array(slice, dtype=int)

    def __len__(self):
        return self.frame_slice.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        decision = False if np.sum(self.sda_class[idx]) < self.sda_slice else True

        return {'in': self.frame_slice[idx], 'sda': self.sda_class[idx], 'out': decision}


def prepare_plotting(data_plot: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Getting data from DataLoader for Plotting Results"""
    din = []
    dsda = []
    dout = []
    for idx, vdata in enumerate(data_plot):
        din = vdata['in'] if idx == 0 else np.append(din, vdata['in'], axis=0)
        dsda = vdata['sda'] if idx == 0 else np.append(dsda, vdata['sda'], axis=0)
        dout = vdata['out'] if idx == 0 else np.append(dout, vdata['out'], axis=0)

    return din, dsda, dout
