import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
from denspp.offline.dnn import DatasetFromFile


class DatasetClassifier(Dataset):
    def __init__(self, dataset: DatasetFromFile):
        """Dataset Loader for Classification Tasks
        :param dataset: Dataclass DatasetFromFile with ['data', 'label', 'names', 'mean']
        :return:        Dataclass Dataset used in PyTorch Training Routine
        """
        self.__data = np.array(dataset.data, dtype=np.float32)
        self.__label = np.array(dataset.label, dtype=np.uint8)
        self.__name = dataset.dict if isinstance(dataset.dict, list) else []

    def __len__(self):
        return self.__data.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        return {
            'in': self.__data[idx,:],
            'out': self.__label[idx]
        }

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return self.__name

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used deep learning topology"""
        return 'Classifier'

    @property
    def get_cluster_num(self) -> int:
        """Getting the number of clusters"""
        return int(np.unique(self.__label).size)


class DatasetAutoencoderClassifier(Dataset):
    def __init__(self, dataset: DatasetFromFile, frames_feat: np.ndarray):
        """Dataset Preparation for training autoencoder-based classifications
        :param dataset:     Dataclass DatasetFromFile with ['data', 'label', 'names', 'mean']
        :param frames_feat: Numpy array with feature space from Encoder output
        :return:            Dataclass Dataset used in PyTorch Training Routine
        """
        self.__data = np.array(dataset.data, dtype=np.float32)
        self.__feat = np.array(frames_feat, dtype=np.float32)
        self.__label = np.array(dataset.label, dtype=np.uint8)
        self.__mean = np.array(dataset.mean, dtype=np.float32)
        self.__labeled_dictionary = dataset.dict if isinstance(dataset.dict, list) else []

    def __len__(self):
        return self.__label.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        return {
            'in': self.__feat[idx, :],
            'out': self.__label[idx]
        }

    @property
    def get_mean_waveforms(self) -> np.ndarray:
        """Getting the mean waveforms of dataset"""
        return self.__mean

    @property
    def get_cluster_num(self) -> int:
        """Returns the number of clusters"""
        return int(np.unique(self.__label).size)

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return self.__labeled_dictionary

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return "Autoencoder-based Classification"
