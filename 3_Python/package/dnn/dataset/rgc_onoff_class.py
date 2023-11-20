import numpy as np
from datetime import datetime
from scipy.io import loadmat
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.data_augmentation import augmentation_reducing_samples
from package.dnn.data_preprocessing import data_normalization


class DatasetRGC(Dataset):
    """Dataset Loader for Retinal Ganglion Cells ON-/OFF Cell Classification"""
    def __init__(self, frame: np.ndarray, cluster_id: np.ndarray, cluster_dict=[]):
        self.__frame_input = np.array(frame, dtype=np.float32)
        self.__frame_cellid = np.array(cluster_id, dtype=np.uint8)
        self.cluster_name_available = False if len(cluster_dict) == 0 else True
        self.frame_dict = cluster_dict
        self.data_type = 'RGC Classification'

    def __len__(self):
        return self.__frame_input.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        out = {'in': self.__frame_input[idx], 'out': self.__frame_cellid[idx]} if not self.cluster_name_available \
            else {'in': self.__frame_input[idx], 'out': self.__frame_cellid[idx], 'name': self.frame_dict[self.__frame_cellid[idx]]}
        return out


def prepare_plotting(data_in: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Getting data from DataLoader for Plotting Results"""
    din = None
    dout = None
    first_run = True
    for vdata in data_in:
        for data in vdata:
            din = data['in'] if first_run else np.append(din, data['in'], axis=0)
            dout = data['out'] if first_run else np.append(dout, data['out'], axis=0)
            first_run = False

    return din, dout


def prepare_training(path: str, settings: Config_PyTorch) -> DatasetRGC:
    """Preparing datasets incl. augmentation for spike-detection-based training (without pre-processing)"""
    print("... loading the datasets")

    # --- MATLAB reading file
    npzfile = loadmat(path)
    frames_in = npzfile["frames_in"]
    frames_cl = npzfile["frames_cluster"].flatten()
    frames_dict = [] # npzfile['cluster_dict']

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

    return DatasetRGC(frames_in, frames_cl, frames_dict)
