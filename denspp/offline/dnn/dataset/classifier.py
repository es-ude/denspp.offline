import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset


class DatasetClassifier(Dataset):
    def __init__(self, frame: np.ndarray, label: np.ndarray, class_name=None):
        """Dataset Loader for Classification Tasks
        Args:
            frame:      Numpy array with all frames
            label:      Numpy array with corresponding label
            class_name: Corresponding dictionary with id label (optional)
        """
        self.__frame_input = np.array(frame, dtype=np.float32)
        self.__frame_cellid = np.array(label, dtype=np.uint8)
        self.__labeled_dictionary = class_name if isinstance(class_name, list) else []

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
        """Getting the information of used deep learning topology"""
        return 'Classification'

    @property
    def get_cluster_num(self) -> int:
        """Getting the number of clusters"""
        return int(np.unique(self.__frame_cellid).size)


def prepare_training(rawdata: dict) -> DatasetClassifier:
    """Generating a dataset class to train a classification model
    :param rawdata:     Dictionary with rawdata for training with labels ['data', 'label', 'dict']
    :return:            Dataloader for Classification task
    """
    frames_in = rawdata['data']
    frames_cl = rawdata['label']
    frames_dict = rawdata['dict']

    check = np.unique(frames_cl, return_counts=True)
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")
    print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if len(frames_dict) == 0 else f' ({frames_dict[idx]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetClassifier(
        frame=frames_in,
        label=frames_cl,
        class_name=frames_dict
    )
