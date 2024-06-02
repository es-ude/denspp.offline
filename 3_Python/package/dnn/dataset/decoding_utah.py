import os

from torch import is_tensor
from torch.utils.data import Dataset

from package.dnn.pytorch_handler import Config_Dataset
from package.dnn.data_augmentation_frames import *


class DatasetDecoder(Dataset):
    """Dataset Preparation for Training Neural Decoder"""
    def __init__(self, spike_train: list, classification: list,
                 cluster_dict: dict, use_patient_dec=True):
        self.__input = spike_train
        self.__output = classification
        self.__use_patient_dec = use_patient_dec

        self.data_type = "Neural Decoder (Utah)"
        self.cluster_name_available = True
        self.frame_dict = cluster_dict

    def __len__(self):
        return len(self.__input)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        if self.__use_patient_dec:
            decision = self.__output[idx]['patient_says']
        else:
            decision = self.__output[idx]['exp_says']

        output = -1
        for key in self.frame_dict.keys():
            if key in decision:
                output = self.frame_dict.get(decision)

        return {'in': np.array(self.__input[idx], dtype=np.float32), 'out': output}


def __generate_stream_empty_array(events: list, cluster: list,
                                  samples_time_window: int,
                                  use_cluster=False,
                                  use_output_size=0) -> np.ndarray:
    """Generating an empty array of the transient array of all electrodes
    Args:
        events: Lists with all timestamps of each electrode (iteration over electrode)
        cluster: Lists with all corresponding cluster unit of each timestamp
        samples_time_window: Size of the window for determining features
        use_cluster: Decision of cluster information will be used
        use_output_size: Determined the output array with a given length
    Return:
        Zero numpy array for training neural decoding [num. electrodes x num. clusters x num. windows] (Starting clusters with Zero)
    """
    length_time_window = np.zeros((len(events, )), dtype=np.uint32)
    num_clusters = np.zeros((len(events, )), dtype=np.uint32)
    for idx, event_ch in enumerate(events):
        length_time_window[idx] = 0 if len(event_ch) == 0 else event_ch[-1]
        num_clusters[idx] = 0 if len(event_ch) == 0 else np.unique(np.array(cluster[idx])).max()+1

    if use_output_size == 0:
        num_windows = int(1 + np.ceil(length_time_window.max() / samples_time_window))
    else:
        num_windows = int(1 + np.ceil(use_output_size / samples_time_window))
    return np.zeros((len(events), num_clusters.max() if use_cluster else 1, num_windows), dtype=np.uint16)


def __determine_firing_rate(events: list, cluster: list, samples_time_window: int,
                            use_cluster=False, use_output_size=0) -> np.ndarray:
    """Pre-Processing Method: Calculating the firing rate for specific
    Args:
        events: Lists with all timestamps of each electrode (iteration over electrode)
        cluster: Lists with all corresponding cluster unit of each timestamp
        samples_time_window: Size of the window for determining features
        use_cluster: Decision of cluster information will be used
        use_output_size: Determined the output array with a given length
    Return:
        Numpy array with windowed number of detected events for training neural decoding [num. electrodes x num. clusters x num. windows]
    """
    data_stream0 = __generate_stream_empty_array(events, cluster, samples_time_window, use_cluster, use_output_size)
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


def translate_ts_datastream_into_picture(data_raw: list, configuration: dict) -> list:
    """ Translate timestamp data stream into picture format """
    picture_data_raw = []
    picture_data_point = None
    labels = configuration['label']

    for data_point in data_raw:
        for elecID, data in enumerate(data_point):
            if picture_data_point is None:
                picture_data_point = np.zeros((data.shape[0], 10, 10, data.shape[1]), dtype=np.uint16)

            for label in labels:
                if f"elec{elecID + 1}" == label:
                    row = configuration['row'][95 - elecID]
                    col = configuration['col'][95 - elecID]
                    picture_data_point[:, col, row, :] = data

        picture_data_raw.append(picture_data_point)
        picture_data_point = None

    return picture_data_raw


def translate_wf_datastream_into_picture(data_raw: list, configuration: dict) -> list:
    """ Translate waveform data stream into picture format"""
    picture_data_raw = []
    labels = configuration['label']
    picture_data_point = [[[] for _ in range(10)] for _ in range(10)]  # array of empty lists

    for data_point in data_raw:
        for elecID, data in enumerate(data_point):
            for label in labels:
                if f"elec{elecID + 1}" == label:  # electrodes from data start at 0, but in mapping at 1.
                    row = configuration['row'][95 - elecID] # col, row range from 0-9 and point to the "correct" label from the mapping file
                    col = configuration['col'][95 - elecID]
                    picture_data_point[col][row] = data  # set list of lists as our list at col,row

        picture_data_raw.append(picture_data_point)
        picture_data_point = [[[] for _ in range(10)] for _ in range(10)]  # after each picture_data_point has been added reset array

    return picture_data_raw


def prepare_training(settings: Config_Dataset,
                     length_time_window_ms=500, use_cluster=False,
                     use_cell_bib=False, mode_classes=2) -> DatasetDecoder:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
    # Construct the full path
    full_path = settings.get_path2data()

    print(f"Constructed Path: {full_path}")
    print(f"Path Exists: {os.path.exists(full_path)}")
    #if os.path.exists(full_path) == False:
        #stop TRAINING
    try:
        data = np.load(full_path)
        print("File loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    data_raw = np.load(settings.get_path2data(), allow_pickle=True).item()

    # --- Pre-Processing: Determine max. timepoint of events
    max_value_timepoint = 0
    for _, data_exp in data_raw.items():
        for key, data_trial in data_exp.items():
            if 'trial' in key:
                events = data_trial['timestamps']

                for event_element in events:
                    if len(event_element):
                        max_value_timepoint = max_value_timepoint if max(event_element) < max_value_timepoint else max(event_element)

    # --- Pre-Processing: Event -> Transient signal transformation
    electrode_mapping = None
    dataset_timestamps = list()
    dataset_decision = list()
    dataset_waveform = list()

    num_ite_skipped = 0
    for _, data_exp in data_raw.items():
        if electrode_mapping is None:
            electrode_mapping = data_exp['orientation']

        for key, data_trial in data_exp.items():
            if 'trial_' in key and not isinstance(data_trial['label']['patient_says'], np.ndarray):
                events = data_trial['timestamps']
                cluster = data_trial['cluster']
                samples_time_window = int(1e-3 * data_trial['samplingrate'] * length_time_window_ms)

                data_stream = __determine_firing_rate(events, cluster, samples_time_window, use_cluster, max_value_timepoint)

                dataset_timestamps.append(data_stream)
                dataset_decision.append(data_trial['label'])
                dataset_waveform.append(data_trial['waveforms'])
            else:
                num_ite_skipped += 1

    del data_exp, data_trial, data_stream, key, events, cluster, samples_time_window

    # --- Pre-Processing: Mapping electrode to 2D-placement
    dataset_timestamps0 = translate_ts_datastream_into_picture(dataset_timestamps, electrode_mapping)
    dataset_waveform0 = translate_wf_datastream_into_picture(dataset_waveform, electrode_mapping)

    # --- Creating dictionary with numbers
    label_dict = dict()
    num_label_types = 0
    for label in dataset_decision:
        used_label = label['patient_says']
        if isinstance(used_label, str):
            if used_label not in label_dict.keys():
                label_dict.update({used_label: num_label_types})
                num_label_types += 1
    del num_label_types, label, used_label

    # --- Counting dataset
    label_count_label_free = [0 for _ in label_dict]
    label_count_label_made = [0 for _ in label_dict]
    for label in dataset_decision:
        used_label = label['patient_says']
        if isinstance(used_label, str):
            idx = label_dict.get(used_label)
            if label['decision'] == 'freeChoice':
                label_count_label_free[idx] += 1
            else:
                label_count_label_made[idx] += 1

    # --- Output
    num_samples = sum(label_count_label_free) + sum(label_count_label_made)
    print(f'... for training are in total {len(label_dict)} classes with {num_samples} samples available')
    if num_ite_skipped:
        print(f"... for training {num_ite_skipped} samples are skipped due to wrong decision values")
    for idx, label in enumerate(label_dict):
        print(f"\t class {idx} ({label}) --> {label_count_label_made[idx] + label_count_label_free[idx]} samples")

    return DatasetDecoder(spike_train=dataset_timestamps0, classification=dataset_decision,
                          cluster_dict=label_dict, use_patient_dec=True)
