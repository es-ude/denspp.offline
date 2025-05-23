import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader


class DatasetSDA(Dataset):
    """Dataset Preparator for training Spike Detection Classification with Neural Network"""
    def __init__(self, frame: np.ndarray, sda: np.ndarray, threshold: int):
        self.__frame_slice = np.array(frame, dtype=np.float32)
        self.__sda_class = np.array(sda, dtype=bool)
        self.__sda_thr = threshold

    def __len__(self):
        return self.__frame_slice.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        decision = 0 if np.sum(self.__sda_class[idx]) < self.__sda_thr else 1

        return {'in': self.__frame_slice[idx], 'sda': self.__sda_class[idx],
                'out': np.array(decision, dtype=np.uint8)}

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled inputs"""
        return ['Non-Spike', 'Spike']

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return 'Spike Detection Algorithm'

    @property
    def get_cluster_num(self) -> int:
        """"""
        return int(np.unique(self.__sda_class).size)


def prepare_training(rawdata: dict, threshold: int) -> DatasetSDA:
    """Preparing datasets incl. augmentation for spike-detection-based training (without pre-processing)"""
    frames_in = rawdata["data"]
    frames_cl = rawdata["label"]

    check = np.unique(frames_cl, return_counts=True)
    print(f"... for training are {frames_in.shape[0]} frames with each {frames_in.shape[1]} points available")
    print(f"... used data points for training: class = {check[0]} and num = {check[1]}")

    return DatasetSDA(
        frame=frames_in,
        sda=frames_cl,
        threshold=threshold
    )
