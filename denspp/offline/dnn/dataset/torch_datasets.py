import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
from denspp.offline.dnn.pytorch_config_data import SettingsDataset


class DatasetTorchVision(Dataset):
    def __init__(self, picture: np.ndarray, label: np.ndarray,
                 cluster_list: list=(), do_classification: bool=False) -> None:
        """Dataset Preparation for training Deep Learning Model using pre-defined datasets from torchvision.datasets
        :param picture:             Numpy data with images to be preprocessed
        :param label:               Numpy data with labels corresponding to the picture
        :param cluster_list:        List of cluster labels corresponding to the labels
        :param do_classification:   Boolean for doing classification (True) or autoencoder (False)
        :return:                    None
        """

        # --- Input Parameters
        self.__frames_orig = np.array(picture, dtype=np.float32)
        self.__frames_size = picture.shape[1]
        self.__cluster_id = np.array(label, dtype=np.uint8)
        self.__do_classification = do_classification
        # --- Parameters for Confusion Matrix for Classification
        self.__labeled_dictionary = cluster_list

    def __len__(self):
        return self.__cluster_id.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.__cluster_id[idx]
        frame_in = self.__frames_orig[idx, :]
        frame_out = self.__frames_orig[idx, :] if not self.__do_classification else cluster_id
        return {'in': frame_in, 'out': frame_out, 'class': cluster_id}

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return self.__labeled_dictionary

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return "MNIST" + (" (Classification)" if self.__do_classification else " (Autoencoder)")

    @property
    def get_cluster_num(self) -> int:
        """Getting the number of classes"""
        return int(np.unique(self.__cluster_id).size)


def prepare_training(rawdata: dict, do_classification: bool) -> DatasetTorchVision:
    """Loading and preparing any dataset for training Deep Learning models from torchvision.datasets
    Args:
        rawdata:            Dictionary with rawdata for training with labels ['data', 'label', 'dict']
        do_classification:  Option for doing a classification, otherwise Autoencoder
    Returns:
        Getting the prepared Dataset
    """
    data_raw = rawdata['data']
    data_dict = rawdata['dict']
    data_label = rawdata['label']

    # --- Print Output
    check = np.unique(data_label, return_counts=True)
    print(f"... for training are {data_raw.shape[0]} frames with each "
          f"({data_raw.shape[1]}, {data_raw.shape[2]}) points available")
    print(f"... used data points for training: "
          f"in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if not isinstance(data_dict, list) else f' ({data_dict[idx]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")
    return DatasetTorchVision(
        picture=data_raw,
        label=data_label,
        cluster_list=data_dict,
        do_classification=do_classification
    )
