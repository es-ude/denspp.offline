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
                                  samples_time_window: int, use_cluster=False) -> np.ndarray:
    """Generating an empty array of the transient array of all electrodes
    Args:
        events: Lists with all timestamps of each electrode (iteration over electrode)
        cluster: Lists with all corresponding cluster unit of each timestamp
        samples_time_window: Size of the window for determining features
        use_cluster: Decision of cluster information will be used
    Return:
        Zero numpy array for training neural decoding [num. electrodes x num. clusters x num. windows] (Starting clusters with Zero)
    """
    length_time_window = np.zeros((len(events, )), dtype=np.uint32)
    num_clusters = np.zeros((len(events, )), dtype=np.uint32)
    for idx, event_ch in enumerate(events):
        length_time_window[idx] = 0 if len(event_ch) == 0 else event_ch[-1]
        num_clusters[idx] = 0 if len(event_ch) == 0 else np.unique(np.array(cluster[idx])).max()+1

    num_windows = int(1 + np.ceil(length_time_window.max() / samples_time_window))
    return np.zeros((len(events), num_clusters.max() if use_cluster else 1, num_windows), dtype=np.uint16)


def __determine_firing_rate(events: list, cluster: list, samples_time_window: int, use_cluster=False) -> np.ndarray:
    """Pre-Processing Method: Calculating the firing rate for specific
    Args:
        events: Lists with all timestamps of each electrode (iteration over electrode)
        cluster: Lists with all corresponding cluster unit of each timestamp
        samples_time_window: Size of the window for determining features
        use_cluster: Decision of cluster information will be used
    Return:
        Numpy array with windowed number of detected events for training neural decoding [num. electrodes x num. clusters x num. windows]
    """
    data_stream0 = __generate_stream_empty_array(events, cluster, samples_time_window, use_cluster)
    for idx, ch_event in enumerate(events):
        if len(ch_event) == 0:
            # Skip due to empty electrode events
            continue
        else:
            # "Slicing" the timestamps of choicen electrode
            ch_event0 = np.array(ch_event)

            if use_cluster:
                ch_cluster = np.array(cluster[idx])
                for cluster_num in np.unique(ch_cluster):
                    sel_event0 = np.argwhere(ch_cluster == cluster_num).flatten()
                    event_ch0 = np.array(np.floor(ch_event0[sel_event0] / samples_time_window), dtype=int)
                    event_val = np.unique(event_ch0, return_counts=True)
                    for idy, pos in enumerate(event_val[0]):
                        data_stream0[idx, cluster_num, pos] += event_val[1][idy]
            else:
                event_ch0 = np.array(np.floor(ch_event0 / samples_time_window), dtype=int)
                event_val = np.unique(event_ch0, return_counts=True)
                for idy, pos in enumerate(event_val[0]):
                    data_stream0[idx, 0, pos] += event_val[1][idy]

    return data_stream0


def prepare_training(settings: Config_Dataset,
                     length_time_window_ms=500, use_cluster=True,
                     use_cell_bib=False, mode_classes=2) -> DatasetDecoder:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
    data_raw = np.load(settings.get_path2data(), allow_pickle=True).item()

    # --- Pre-Processing: Do spike sorting (future content)
    data_sorted = data_raw

    # --- Pre-Processing: Event -> Transient signal transformation
    dataset_timestamps = list()
    dataset_decision = list()
    dataset_waveform = list()
    for _, data_exp in data_sorted.items():
        for key, data_trial in data_exp.items():
            if 'trial_' in key:
                events = data_trial['timestamps']
                cluster = data_trial['cluster']
                samples_time_window = int(1e-3 * data_trial['samplingrate'] * length_time_window_ms)
                data_stream = __determine_firing_rate(events, cluster, samples_time_window, use_cluster)

                # --- Step #3: Transfer result to output
                dataset_timestamps.append(data_stream)
                dataset_decision.append(data_trial['label'])
                dataset_waveform.append(data_trial['waveforms'])

    # ---

    # --- Pre-Processing: Mapping electrode to 2D-placement

    # --- Output
    return DatasetDecoder()
