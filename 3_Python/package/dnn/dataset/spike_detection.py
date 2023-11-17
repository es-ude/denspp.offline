import numpy as np
from datetime import datetime
from scipy.io import loadmat
from torch import is_tensor, Tensor
from torch.utils.data import Dataset, DataLoader
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.data_augmentation import augmentation_reducing_samples
from package.dnn.data_preprocessing import data_normalization


class DatasetSDA(Dataset):
    """Dataset Preparator for training Spike Detection Classification with Neural Network"""
    def __init__(self, frame: np.ndarray, sda: np.ndarray, threshold: int):
        self.frame_slice = np.array(frame, dtype=np.float32)
        self.sda_class = np.array(sda, dtype=bool)
        self.sda_thr = threshold

    def __len__(self):
        return self.frame_slice.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        decision = 0 if np.sum(self.sda_class[idx]) < self.sda_thr else 1

        return {'in': self.frame_slice[idx], 'sda': self.sda_class[idx], 'out': np.array([decision], dtype=np.float32)}


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


def prepare_training(path: str, settings: Config_PyTorch) -> DatasetSDA:
    """Preparing datasets incl. augmentation for spike-detection-based training (without pre-processing)"""
    # --- Pre-definitions
    str_datum = datetime.now().strftime('%Y%m%d %H%M%S')
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    # --- MATLAB reading file
    npzfile = loadmat(path)
    frames_in = npzfile["frames_in"]
    frames_cl = npzfile["frames_cluster"].flatten()
    frames_dict = npzfile['cluster_dict']

    # --- PART: Exclusion of selected clusters
    if not len(settings.data_exclude_cluster) == 0:
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

    # --- PART: Data Normalization
    if settings.data_do_normalization:
        frames_in = data_normalization(frames_in)

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    print(f"... for training are {frames_in.shape[0]} frames with each {frames_in.shape[1]} points available")
    if len(frames_dict) == 0:
        print(f"... used data points for training: class = {check[0]} and num = {check[1]}")
    else:
        print(f"... used data points for training: class = {frames_dict[check[0]]} and num = {check[1]}")

    return DatasetSDA(frames_in, frames_cl, frames_dict)
