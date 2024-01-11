import numpy as np
from scipy.io import loadmat
from torch import is_tensor, randn, Tensor
from torch.utils.data import Dataset

from package.dnn.pytorch_control import Config_Dataset
from package.dnn.data_preprocessing import calculate_frame_snr, calculate_frame_mean, calculate_frame_median
from package.dnn.data_preprocessing import change_frame_size, reconfigure_cluster_with_cell_lib, generate_zero_frames, data_normalization_minmax
from package.dnn.data_augmentation import *


class DatasetDecoder(Dataset):
    """Dataset Preparation for training Autoencoders"""
    def __init__(self, spike_train: np.ndarray, classification: np.ndarray,
                 cluster_dict=None, do_classification=False):

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
                     use_cell_bib=False, mode_classes=2,
                     use_median_for_mean=True,
                     mode_train_ae=0, do_classification=False,
                     noise_std=0.1) -> DatasetDecoder:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
    npzfile = loadmat(settings.get_path2data())
    frames_in = npzfile["frames_in"]
    frames_cl = npzfile["frames_cluster"].flatten() if 'frames_cluster' in npzfile else npzfile["frames_cl"].flatten()
    frames_dict = None

    # --- Using cell_bib for clustering
    if use_cell_bib:
        frames_in, frames_cl, frames_dict = reconfigure_cluster_with_cell_lib(settings.get_path2data(),
                                                                              mode_classes, frames_in, frames_cl)

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")
    print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if not isinstance(frames_dict, list | np.ndarray) else f' ({frames_dict[id]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetAE(frames_raw=frames_in, cluster_id=frames_cl, frames_cluster_me=frames_me,
                     cluster_dict=frames_dict, mode_train=mode_train_ae, do_classification=do_classification,
                     noise_std=noise_std)
