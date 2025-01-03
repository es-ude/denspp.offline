import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
from package.dnn.pytorch_config_data import ConfigDataset


class DatasetMNIST(Dataset):
    """Dataset Preparator for training Autoencoder"""
    def __init__(self, picture: np.ndarray, label: np.ndarray,
                 cluster_dict=None, do_classification=False):

        # --- Input Parameters
        self.__frames_orig = np.array(picture, dtype=np.float32)
        self.__frames_size = picture.shape[1]
        self.__cluster_id = np.array(label, dtype=np.uint8)
        self.__do_classification = do_classification
        # --- Parameters for Confusion Matrix for Classification
        self.__labeled_dictionary = cluster_dict if isinstance(cluster_dict, list) else []

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
        """"""
        return int(np.unique(self.__cluster_id).size)


def prepare_training(settings: ConfigDataset, do_classification=True) -> DatasetMNIST:
    """Loading and preparing the MNIST dataset for Deep Learning
    Args:
        settings:           Class for loading and pre-processing the data for DataLoader
        do_classification:  Option for doing a classification, otherwise Autoencoder
    Returns:
        Getting the prepared Dataset for MNIST
    """
    dataset = settings.load_dataset()
    data_raw = dataset['data']
    data_dict = dataset['dict']
    data_label = dataset['label']

    # --- Normalization
    if settings.normalization_do:
        data_raw = data_raw / 255.0
        print("... do data normalization on input")

    # --- Exclusion of selected clusters
    if len(settings.exclude_cluster):
        for i, id in enumerate(settings.exclude_cluster):
            selX = np.where(data_label != id)
            data_raw = data_raw[selX[0], :]
            data_label = data_label[selX]
        print(f"... class reduction done to {np.unique(data_label).size} classes")

    # --- Using cell library
    if settings.use_cell_library:
        raise NotImplementedError("No cell library for this case is available - Please disable flag!")

    # --- Data Augmentation
    if settings.augmentation_do:
        raise NotImplementedError("No augmentation method is implemented - Please disable flag!")

    if settings.reduce_samples_per_cluster_do:
        raise NotImplementedError(f"No reducing samples technique is implemented - Please disable flag!")

    # --- Print Output
    check = np.unique(data_label, return_counts=True)
    print(f"... for training are {data_raw.shape[0]} frames with each "
          f"({data_raw.shape[1]}, {data_raw.shape[2]}) points available")
    print(f"... used data points for training: "
          f"in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if not isinstance(data_dict, list) else f' ({data_dict[id]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetMNIST(data_raw, data_label, data_dict, do_classification)
