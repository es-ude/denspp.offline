import numpy as np
from scipy.io import loadmat
from torch import is_tensor, randn, Tensor
from torch.utils.data import Dataset

from package.dnn.pytorch_control import Config_Dataset
from package.dnn.data_preprocessing_frames import reconfigure_cluster_with_cell_lib
from package.dnn.data_augmentation_frames import *


class DatasetDecoder(Dataset):
    """Dataset Preparation for training Autoencoders"""
    def __init__(self, spike_train: np.ndarray, classification: np.ndarray, cluster_dict=None):

        self.size_output = 4
        # --- Input Parameters
        self.__input = np.array(spike_train, dtype=np.float32)
        self.__output = np.array(classification, dtype=np.float32)
        self.cluster_name_available = isinstance(cluster_dict, list)
        self.frame_dict = cluster_dict

    def __len__(self):
        return self.__output.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.__output[idx]
        return {'in': self.__input[idx, :],
                'out': self.__output[idx, :],
                'class': cluster_id}


def prepare_training(settings: Config_Dataset,
                     use_cell_bib=False, mode_classes=2) -> DatasetDecoder:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
    rawdata = np.load(settings.get_path2data(), allow_pickle=True).item()
    data_exp = rawdata['exp_000']
    print('TEST')

    # --- Output
    return DatasetDecoder()
