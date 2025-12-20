import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
from denspp.offline.dnn import DatasetFromFile


class DatasetAutoencoder(Dataset):
    def __init__(self, dataset: DatasetFromFile, noise_std=0.1, mode_train=0):
        """Dataset Preparator for training Autoencoder
        :param dataset:             Dataclass DatasetFromFile with data from extern:
        :param noise_std:           Adding noise standard deviation on input data
        :param mode_train:          Autoencoder Training Mode
                                    [0: Autoencoder,
                                    1: Denoising Autoencoder (mean),
                                    2: Denoising Autoencoder (add random noise),
                                    3: Denoising Autoencoder (add gaussian noise)]
        """
        self.__mode = ["", "(mean) Denoising ", "(random noise) Denoising ", "(gaussian noise) Denoising "]
        self.__noise_std = noise_std
        self.__mode_train = mode_train

        self.__data = np.array(dataset.data, dtype=np.float32)
        self.__size = dataset.data.shape[1:]
        self.__label = np.array(dataset.label, dtype=np.uint8)
        self.__mean = np.array(dataset.mean, dtype=np.float32)
        self.__labeled_dictionary = dataset.dict if isinstance(dataset.dict, list) else []

    def __len__(self):
        return self.__label.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.__label[idx]
        if self.__mode_train == 1:
            # Denoising Autoencoder Training with mean
            frame_in = self.__data[idx, :]
            frame_out = self.__mean[cluster_id, :]
        elif self.__mode_train == 2:
            # Denoising Autoencoder Training with adding random noise on input
            frame_in = self.__data[idx, :] + np.array(self.__noise_std * np.random.randn(*self.__size), dtype=np.float32)
            frame_out = self.__data[idx, :]
        elif self.__mode_train == 3:
            # Denoising Autoencoder Training with adding gaussian noise on input
            frame_in = self.__data[idx, :] + np.array(self.__noise_std * np.random.normal(size=self.__size), dtype=np.float32)
            frame_out = self.__data[idx, :]
        else:
            # Normal Autoencoder Training
            frame_in = self.__data[idx, :]
            frame_out = self.__data[idx, :]
        return {
            'in': frame_in,
            'out': frame_out,
            'class': cluster_id,
            'mean': self.__mean[cluster_id, :]
        }

    @property
    def get_mean_waveforms(self) -> np.ndarray:
        """Getting the mean waveforms of dataset"""
        return self.__mean

    @property
    def get_cluster_num(self) -> int:
        """Returning the number of unique classes/labels in the dataset"""
        return int(np.unique(self.__label).size)

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return self.__labeled_dictionary

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return self.__mode[self.__mode_train] + "Autoencoder"
