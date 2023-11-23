import numpy as np
from scipy.io import loadmat
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from package.data.data_call_addon import CellSelector
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.data_augmentation import augmentation_reducing_samples
from package.dnn.data_preprocessing import data_normalization


class DatasetRGC(Dataset):
    """Dataset Loader for Retinal Ganglion Cells ON-/OFF Cell Classification"""
    def __init__(self, frame: np.ndarray, cluster_id: np.ndarray, cluster_dict=None):
        self.__frame_input = np.array(frame, dtype=np.float32)
        self.__frame_cellid = np.array(cluster_id, dtype=np.uint8)
        self.cluster_name_available = isinstance(cluster_dict, list)
        self.frame_dict = cluster_dict
        self.data_type = 'RGC Classification'

    def __len__(self):
        return self.__frame_input.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        return {'in': self.__frame_input[idx], 'out': self.__frame_cellid[idx]}


def prepare_training(path: str, settings: Config_PyTorch, use_cell_bib=False, mode_classes=0) -> DatasetRGC:
    """Preparing datasets incl. augmentation for spike-detection-based training (without pre-processing)"""
    print("... loading the datasets")

    # --- MATLAB reading file
    npzfile = loadmat(path)
    frames_in = npzfile["frames_in"]
    frames_cl = npzfile["frames_cluster"].flatten()
    frames_dict = npzfile['cluster_dict'].tolist()

    # --- PART: Exclusion of selected clusters
    if not len(settings.data_exclude_cluster) == 0:
        for i, id in enumerate(settings.data_exclude_cluster):
            selX = np.where(frames_cl != id)
            frames_in = frames_in[selX[0], :]
            frames_cl = frames_cl[selX]

    # --- PART: Using a cell bib with option to reduce cluster
    if use_cell_bib:
        frames_cl, frames_dict = __reducing_cluster_samples(path, frames_cl, mode_classes)

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
    if len(frames_dict) == 0:
        print(f"... used data points for training: class = {check[0]} and num = {check[1]}")
    else:
        print(f"... used data points for training: class = {frames_dict} and num = {check[1]}")

    return DatasetRGC(frames_in, frames_cl, frames_dict)


def __reducing_cluster_samples(path: str, frames_cl: np.ndarray, sel_mode_classes: int) -> [np.ndarray, dict]:
    """Function for reducing the samples for a given cell bib"""
    cell_dict = list()
    cell_cl = frames_cl

    check_class = ['fzj', 'RGC']
    check_path = path[:-4].split("_")
    # --- Check if one is available
    flag = -1
    for path0 in check_path:
        for idx, j in enumerate(check_class):
            if path0 == j:
                flag = idx
                break

    if not flag == -1:
        print("... reducing output classes")
        cl_sampler = CellSelector(flag, sel_mode_classes)
        cell_dict = cl_sampler.get_classes()
        for idx, cl in enumerate(frames_cl):
            frames_cl[idx] = cl_sampler.get_class_to_id(cl)[0]

    return cell_cl, cell_dict
