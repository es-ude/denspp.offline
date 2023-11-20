import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetClass(Dataset):
    """Dataset Preparator for training Classification Neural Network"""
    def __init__(self, frames: np.ndarray, feat: np.ndarray, index: np.ndarray, type=''):
        self.__frames_orig = np.array(frames, dtype=np.float32)
        self.__frames_feat = np.array(feat, dtype=np.float32)
        self.__frames_clus = index
        self.data_type = 'Classifier' if not type else type

    def __len__(self):
        return self.__frames_clus.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_orig = self.__frames_orig[idx, :]
        frame_feat = self.__frames_feat[idx, :]
        frame_clus = self.__frames_clus[idx]

        return {'in': frame_orig, 'feat': frame_feat, 'cluster': frame_clus}


def prepare_plotting(data_plot: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Getting data from DataLoader for Plotting Results"""
    din = []
    dout = []
    did = []
    dmean = []
    for i, vdata in enumerate(data_plot):
        din0 = vdata['in']
        dout0 = vdata['out']
        dmean0 = vdata['mean']
        did0 = vdata['cluster']

        din = din0 if i == 0 else np.append(din, din0, axis=0)
        dout = dout0 if i == 0 else np.append(dout, dout0, axis=0)
        dmean = dmean0 if i == 0 else np.append(dmean, dmean0, axis=0)
        did = did0 if i == 0 else np.append(did, did0)

    return din, dout, did, dmean
