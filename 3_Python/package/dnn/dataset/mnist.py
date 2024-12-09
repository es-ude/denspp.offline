import os
import numpy as np
from torchvision import datasets, transforms
from torch import is_tensor, concat
from torch.utils.data import Dataset
from package.dnn.pytorch_dataclass import Config_Dataset


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


def load_mnist(data_path: str) -> [datasets.MNIST, datasets.MNIST]:
    """Loading MNIST dataset and preparing for training
    Args:
        data_path:  String for finding the MNIST data locally
    Returns:
        Two dataset arrays with [training samples, validation samples]
    """
    # --- Checking if dataset exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        do_download = True
    else:
        path2mnist = os.path.join(data_path, 'MNIST')
        if not os.path.exists(path2mnist):
            do_download = True
        else:
            do_download = False

    # --- Resampling of MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    data_train = datasets.MNIST(data_path, train=True, download=do_download, transform=transform)
    data_valid = datasets.MNIST(data_path, train=False, download=do_download, transform=transform)
    return data_train, data_valid


def prepare_training(settings: Config_Dataset, do_classification=True) -> DatasetMNIST:
    """Loading and preparing the MNIST dataset for Deep Learning
    Args:
        settings:           Class for loading and pre-processing the data for DataLoader
        do_classification:  Option for doing a classification, otherwise Autoencoder
    Returns:
        Getting the prepared Dataset for MNIST
    """
    data_train, data_valid = load_mnist(settings.get_path2folder_data)

    # --- Translating data to common
    data_raw = concat((data_train.data, data_valid.data), 0).numpy()
    data_label = concat((data_train.targets, data_valid.targets), 0).numpy()
    data_dict = data_train.classes

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
