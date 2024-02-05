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


def __generate_stream_empty_array(events: list, cluster: list,
                            fs: int, length_time_window_ms: float, use_cluster=False) -> np.ndarray:
    length_time_window = np.zeros((len(events, )), dtype=np.uint32)
    num_clusters = np.zeros((len(events, )), dtype=np.uint32)
    for idx, event_ch in enumerate(events):
        length_time_window[idx] = 0 if len(event_ch) == 0 else event_ch[-1]
        num_clusters[idx] = 0 if len(event_ch) == 0 else np.unique(np.array(cluster[idx])).size

    dt_time_window = int(1e-3 * fs * length_time_window_ms)
    num_windows = int(1 + np.ceil(length_time_window.max() / dt_time_window))
    return np.zeros((len(events), num_clusters.max() if use_cluster else 1, num_windows), dtype=np.uint16)


def __determine_firing_rate(events: list, cluster: list,
                            fs: int, length_time_window_ms: float, use_cluster=False) -> np.ndarray:
    data_stream0 = __generate_stream_empty_array(events, cluster, fs, length_time_window_ms, use_cluster)
    dt_time_window = int(1e-3 * fs * length_time_window_ms)

    for idx, event_ch in enumerate(events):
        if len(event_ch) == 0:
            # Skip due to empty electrode events
            continue
        else:
            # "Slicing" the timestamps of choicen electrode
            event_ch0 = np.array(np.floor(np.array(event_ch) / dt_time_window), dtype=int)
            data_stream0[idx, 0, event_ch0] += 1

    return data_stream0


def prepare_training(settings: Config_Dataset,
                     use_cell_bib=False, mode_classes=2, length_time_window_ms=10, use_cluster=False) -> DatasetDecoder:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
    data_raw = np.load(settings.get_path2data(), allow_pickle=True).item()

    # --- Pre-Processing: Do spike sorting (future content)
    data_sorted = data_raw

    # --- Translating rawdata into stream data for dataset
    dataset_timestamps = list()
    dataset_decision = list()
    for _, data_exp in data_sorted.items():
        for key, data_trial in data_exp.items():
            if 'trial_' in key:
                events = data_trial['timestamps']
                cluster = data_trial['cluster']

                data_stream = __determine_firing_rate(events, cluster, data_trial['samplingrate'], length_time_window_ms, use_cluster)

                # --- Step #3: Transfer result to output
                dataset_timestamps.append(data_stream)
                dataset_decision.append(data_trial['label'])

    # --- Output
    return DatasetDecoder()
