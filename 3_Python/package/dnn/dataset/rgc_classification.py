import numpy as np
from scipy.io import loadmat
from torch import is_tensor
from torch.utils.data import Dataset
from package.dnn.pytorch_handler import Config_Dataset
from package.data_process.frame_augmentation import augmentation_reducing_samples
from package.data_process.frame_preprocessing import reconfigure_cluster_with_cell_lib
from package.data_process.frame_normalization import DataNormalization


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


def prepare_training(settings: Config_Dataset, use_cell_bib=False, mode_classes=0) -> DatasetRGC:
    """Preparing dataset incl. augmentation for spike-detection-based training"""
    print("... loading and processing the dataset")
    npzfile = loadmat(settings.get_path2data())
    frames_in = npzfile["frames_in"]
    frames_cl = npzfile["frames_cluster"].flatten() if 'frames_cluster' in npzfile else npzfile["frames_cl"].flatten()
    frames_dict = dict()

    # --- PART: Exclusion of selected clusters
    if not len(settings.data_exclude_cluster) == 0:
        for i, id in enumerate(settings.data_exclude_cluster):
            selX = np.where(frames_cl != id)
            frames_in = frames_in[selX[0], :]
            frames_cl = frames_cl[selX]

    # --- PART: Using a cell bib with option to reduce cluster
    if use_cell_bib:
        frames_in, frames_cl, frames_dict = reconfigure_cluster_with_cell_lib(settings.get_path2data(),
                                                                              mode_classes, frames_in, frames_cl)

    # --- PART: Data Normalization
    if settings.data_do_normalization:
        if settings.data_normalization_setting == 'bipolar':
            do_bipolar = True
            do_global = False
        elif settings.data_normalization_setting == 'global':
            do_bipolar = False
            do_global = True
        elif settings.data_normalization_setting == 'combined':
            do_bipolar = True
            do_global = True
        else:
            do_bipolar = False
            do_global = False
        print(f"... do data normalization")
        data_class_frames_in = DataNormalization(mode=settings.data_normalization_mode,
                                                 method=settings.data_normalization_method,
                                                 do_bipolar=do_bipolar, do_global=do_global)
        frames_in = data_class_frames_in.normalize(frames_in)

    # --- PART: Reducing samples per cluster (if too large)
    if settings.data_do_reduce_samples_per_cluster:
        print("... reducing the samples per cluster (for pre-training on dedicated hardware)")
        frames_in, frames_cl = augmentation_reducing_samples(frames_in, frames_cl,
                                                             settings.data_num_samples_per_cluster)

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")
    print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if len(frames_dict) == 0 else f' ({frames_dict[id]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetRGC(frames_in, frames_cl, frames_dict)
