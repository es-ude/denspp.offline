import numpy as np
from scipy.io import loadmat
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.data_augmentation import augmentation_reducing_samples
from package.dnn.data_preprocessing import data_normalization


class DatasetSDA(Dataset):
    """Dataset Preparator for training Spike Detection Classification with Neural Network"""
    def __init__(self, frame: np.ndarray, sda: np.ndarray, threshold: int):
        self.__frame_slice = np.array(frame, dtype=np.float32)
        self.__sda_class = np.array(sda, dtype=bool)
        self.__sda_thr = threshold
        self.sda_dict = ['Non-Spike', 'Spike']
        self.data_type = 'Spike Detection Algorithm'

    def __len__(self):
        return self.__frame_slice.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        decision = 0 if np.sum(self.__sda_class[idx]) < self.__sda_thr else 1

        return {'in': self.__frame_slice[idx], 'sda': self.__sda_class[idx],
                'out': np.array(decision, dtype=np.uint8)}


def prepare_plotting(data_in: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Getting data from DataLoader for Plotting Results"""
    din = None
    dsda = None
    dout = None
    first_run = True
    for vdata in data_in:
        for data in vdata:
            din = data['in'] if first_run else np.append(din, data['in'], axis=0)
            dsda = data['sda'] if first_run else np.append(dsda, data['sda'], axis=0)
            dout = data['out'] if first_run else np.append(dout, data['out'], axis=0)
            first_run = False

    return din, dsda, dout


def prepare_training(path: str, settings: Config_PyTorch, threshold: int) -> DatasetSDA:
    """Preparing datasets incl. augmentation for spike-detection-based training (without pre-processing)"""
    print("... loading the datasets")

    # --- MATLAB reading file
    npzfile = loadmat(path)
    frames_in = npzfile["sda_in"]
    frames_cl = npzfile["sda_pred"]

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
    print(f"... used data points for training: class = {check[0]} and num = {check[1]}")

    return DatasetSDA(frames_in, frames_cl, threshold)
