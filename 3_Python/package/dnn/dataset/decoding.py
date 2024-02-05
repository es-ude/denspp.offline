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
                     use_cell_bib=False, mode_classes=2, length_time_window_ms=10, use_cluster=False) -> DatasetDecoder:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
    data_raw = np.load(settings.get_path2data(), allow_pickle=True).item()

    # --- Translating rawdata into stream data for dataset
    dataset_timestamps = list()
    dataset_decision = list()
    for key, data_exp in data_raw.items():
        for _, data_trial in data_exp.items():
            events = data_trial['timestamps']
            cluster = data_trial['cluster']

            # --- Step #1: Generating empty transient array
            length_time_window = np.zeros((len(events, )), dtype=np.uint32)
            num_clusters = np.zeros((len(events, )), dtype=np.uint32)
            for idx, event_ch in enumerate(events):
                length_time_window[idx] = 0 if len(event_ch) == 0 else event_ch[-1]
                num_clusters[idx] = 0 if len(event_ch) == 0 else np.unique(np.array(cluster[idx])).size

            dt_time_window = int(1e-3 * data_trial['samplingrate'] * length_time_window_ms)
            data_stream = np.zeros((len(events), num_clusters.max() if use_cluster else 1, int(1 + np.ceil(length_time_window.max()/dt_time_window))), dtype=np.uint16)

            # --- Step #2: Generating transient signal of firing rate (Pre-Processing)
            print("Test")

            # --- Step #3: Transfer result to output
            dataset_timestamps.append(data_stream)
            dataset_decision.append(data_trial['label'])

    # --- Output
    return DatasetDecoder()
