import numpy as np
from datetime import datetime
from scipy.io import loadmat
from torch import is_tensor, Tensor
from torch.utils.data import Dataset, DataLoader
from package.dnn.pytorch_control import Config_PyTorch


class DatasetRGC(Dataset):
    """Dataset Loader for Retinal Ganglion Cells ON-/OFF Cell Classification"""
    def __init__(self, frame: np.ndarray, sda: np.ndarray):
        self.frame_slice = np.array(frame, dtype=np.float32)
        self.frame_cellid = np.array(sda, dtype=bool)

    def __len__(self):
        return self.frame_slice.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        return {'in': self.frame_slice[idx], 'out': self.frame_cellid[idx]}


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


def prepare_training(path: str, settings: Config_PyTorch) -> DatasetRGC:
    """Preparing datasets incl. augmentation for spike-detection-based training (without pre-processing)"""
    # --- Pre-definitions
    str_datum = datetime.now().strftime('%Y%m%d %H%M%S')
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    # --- MATLAB reading file
    npzfile = loadmat(path)
    frames_in = npzfile["sda_in"]
    frames_pred = npzfile["sda_pred"]
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    return DatasetSDA(frames_in, frames_pred, 3)
