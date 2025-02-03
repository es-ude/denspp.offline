import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
from denspp.offline.dnn.pytorch_handler import ConfigDataset


class DatasetClassifier(Dataset):
    def __init__(self, frame: np.ndarray, cluster_id: np.ndarray, cluster_dict=None):
        """Dataset Loader for Retinal Ganglion Cells ON-/OFF Cell Classification
        Args:
            frame:          Numpy array with all frames
            cluster_id:     Corresponding spike label of each frame
            cluster_dict:   Corresponding dictionary with id label (optional)
        """
        self.__frame_input = np.array(frame, dtype=np.float32)
        self.__frame_cellid = np.array(cluster_id, dtype=np.uint8)
        self.__labeled_dictionary = cluster_dict if isinstance(cluster_dict, list) else []

    def __len__(self):
        return self.__frame_input.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        return {'in': self.__frame_input[idx], 'out': self.__frame_cellid[idx]}

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return self.__labeled_dictionary

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return 'Classification'

    @property
    def get_cluster_num(self) -> int:
        """"""
        return int(np.unique(self.__frame_cellid).size)


def prepare_training(settings: ConfigDataset) -> DatasetClassifier:
    """Preparing dataset incl. augmentation for spike-detection-based training
    Args:
        settings:       Settings for loading data
    Return:
        Dataloader with retinal ganglion cell types for classification tasks
    """
    rawdata = settings.load_dataset()
    frames_in = rawdata['data']
    frames_cl = rawdata['label']
    frames_dict = rawdata['dict']

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")
    print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if len(frames_dict) == 0 else f' ({frames_dict[id]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetClassifier(frames_in, frames_cl, frames_dict)
