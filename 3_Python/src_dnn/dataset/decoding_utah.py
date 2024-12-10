import os
import sys
import csv
import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
from pathlib import Path

from package.dnn.pytorch_handler import Config_Dataset


class DecoderDataset(Dataset):
    def __init__(self, dataset_spike_train: list[dict], decision: list[dict], label_dict: dict, use_patient_dec=True):
        """Dataset Preparation for Training Neural Decoder
        Args:
            dataset_spike_train:    List with firing rate activity
            decision:               List with labeled information
            label_dict:             Dictionary of what kind of labels are available
            use_patient_dec:        Boolean for label extraction (False = Experiment, True = Patient)
        """
        self.__datset_spike_train = dataset_spike_train
        self.__decision = decision
        self.__use_patient_dec = use_patient_dec
        self.__labeled_dictionary = label_dict

    def __len__(self):
        return len(self.__datset_spike_train)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        if self.__use_patient_dec:
            decision = self.__decision[idx]['patient_says']
        else:
            decision = self.__decision[idx]['exp_says']

        output = -1
        for key in self.__labeled_dictionary.keys():
            if key in decision:
                output = self.__labeled_dictionary.get(decision)

        return {'in': np.array(self.__datset_spike_train[idx], dtype=np.float32), 'out': output}

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return [key for key in self.__labeled_dictionary.keys()]

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return "Neural Decoder (Utah)"


def prepare_training(settings: Config_Dataset, length_time_window_ms=500, use_cluster=True) -> DecoderDataset:
    """Preparing dataset incl. add time-window feature and counting dataset
    Args:
        settings:               Configuration/Settings for handling data
        length_time_window_ms   Time window for looking on neural activity (firing rate)
        use_cluster:            Using the cluster results (spike-sorted) for more-detailed classification
    Returns:
        Modified DataLoader for Neural Decoding of Movement Ambitions using Utah Array
    """
    data_raw = __load_dataset(settings)
    max_overall_timestamp = __get_max_timestamp(data_raw)
    (dataset_decision, dataset_timestamps,
     dataset_waveform, num_ite_skipped) = __create_feature_dataset(
        data_raw=data_raw,
        length_time_window_ms=length_time_window_ms,
        max_overall_timestamp=max_overall_timestamp,
        use_cluster=use_cluster
    )
    # --- Pre-Processing: Mapping electrode to 2D-placement
    dataset_spike_train = __add_electrode_2d_mapping_to_dataset(data_raw, dataset_timestamps, dataset_waveform)

    # --- Creating lable dictionary
    label_dict = __create_label_dict(dataset_decision)

    # --- Counting dataset
    label_count_label_free, label_count_label_made, num_samples = __counting_dataset(dataset_decision, label_dict)

    # --- Using cell library
    if settings.use_cell_library:
        raise NotImplementedError("No cell library for this case is available - Please disable flag!")

    # --- Do normalization method
    if settings.normalization_do:
        raise NotImplementedError("No normalization method is implemented - Please disable flag!")

    # Do data augmentation
    if settings.augmentation_do:
        raise NotImplementedError("No data augmentation method is implemented - Please disable flag!")
    if settings.reduce_samples_per_cluster_do:
        raise NotImplementedError("No reducing sample technique is implemented - Please disable flag!")
    if len(settings.exclude_cluster):
        raise NotImplementedError("No class excluding method is implemented - Please remove content!")

    # --- Console Output
    print(f'\t for training are in total {len(label_dict)} classes with {num_samples} samples available')
    if num_ite_skipped:
        print(f"\t for training {num_ite_skipped} samples are skipped due to wrong decision values")
    for idx, label in enumerate(label_dict):
        print(f"\t\t class {idx} ({label}) --> {label_count_label_made[idx] + label_count_label_free[idx]} samples")

    return DecoderDataset(dataset_spike_train=dataset_spike_train, decision=dataset_decision,
                          label_dict=label_dict, use_patient_dec=True)


def generate_electrode_mapping_from_data(settings: Config_Dataset, path2save_csv='') -> np.ndarray:
    """Preparing dataset incl. add time-window feature and counting dataset
    Args:
        settings:               Configuration/Settings for handling data
        path2save_csv:          Path for saving the mapping in *.csv
    Returns:
        Numpy array with electrode ID mapping
    """
    # --- Get electrode mapping information
    data_raw = __load_dataset(settings)
    electrode_mapping_orig = data_raw['exp_000']['orientation']
    labels = electrode_mapping_orig['label']

    # --- Generate electrode mapping
    elec_id_map = np.zeros((10, 10), dtype=int)
    for id, label in enumerate(labels):
        elec_id = int(label.split('elec')[1])  # elec bezieht sich auf den Datensatz nicht ändern!!
        row = electrode_mapping_orig['row'][95 - id]
        col = electrode_mapping_orig['col'][95 - id]
        elec_id_map[col, row] = elec_id

    # --- Write csv file
    if path2save_csv:
        with open(os.path.join(path2save_csv, 'Mapping_Utah.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in elec_id_map:
                writer.writerow([id for id in row])
    return elec_id_map


def __load_dataset(settings: Config_Dataset) -> np.ndarray:
    """Loading the raw data from Utah recording"""
    base_path = Path(__file__).parents[2]
    func_name = __load_dataset.__name__
    # Pfad ab dem Ordner "3_Python" extrahieren
    shortened_path = Path(__file__).relative_to(base_path)
    print(
        f"\n\n=== Executing function --> {func_name} in file --> {shortened_path} === ")
    print("\n\t loading and preprocessing the dataset")

    # Construct the full path of the dataset
    full_path = settings.get_path2data
    if not os.path.exists(full_path):
        print("\n File Path not right")
        sys.exit(1)
    print(f"\t Constructed Path: {full_path}")
    data_raw = np.load(settings.get_path2data, allow_pickle=True).item()
    return data_raw


def __get_max_timestamp(data_raw) -> int:
    """"""
    max_overall_timestamp = 0
    for _, data_exp in data_raw.items():  # 0-20 experiment
        for key, data_trial in data_exp.items():
            if 'trial' in key:
                trial_timestamps = data_trial['timestamps']
                for timestamp_in_electrode in trial_timestamps:
                    if len(timestamp_in_electrode) and max(timestamp_in_electrode) > max_overall_timestamp:
                        max_overall_timestamp = max(timestamp_in_electrode)
    return max_overall_timestamp


def __create_feature_dataset(data_raw, length_time_window_ms, max_overall_timestamp, use_cluster):
    """"""
    # --- Pre-Processing: Event --> Transient signal transformation
    dataset_timestamps = list()
    dataset_decision = list()
    dataset_waveform = list()
    num_ite_skipped = 0

    exp_samplingrate_in_hertz = data_raw['exp_000']['trial_000']['samplingrate']
    for _, data_exp in data_raw.items():

        for key, data_trial in data_exp.items():
            if 'trial_' in key and not isinstance(data_trial['label']['patient_says'], np.ndarray):
                timestamps = data_trial['timestamps']
                cluster_per_trial = data_trial['cluster']

                timestamp_stream = __determine_firing_rate(timestamps,
                                                           cluster_per_trial,
                                                           exp_samplingrate_in_hertz,
                                                           length_time_window_ms,
                                                           use_cluster,
                                                           max_overall_timestamp
                                                           )

                dataset_timestamps.append(timestamp_stream)
                dataset_decision.append(data_trial['label'])
                dataset_waveform.append(data_trial['waveforms'])
            else:
                num_ite_skipped += 1
    return dataset_decision, dataset_timestamps, dataset_waveform, num_ite_skipped


def __generate_stream_empty_array(timestamps: list, cluster: list,
                                  samples_time_window: int,
                                  use_cluster=True,
                                  output_size=0) -> np.ndarray:
    """Generating an empty array of the transient array of all electrodes
    Args:
        timestamps:             Lists with all timestamps of each electrode (iteration over electrode)
        cluster:                Lists with all corresponding cluster unit of each timestamp
        samples_time_window:    Size of the window for determining features use_cluster: Decision of cluster information
                                will be used output_size: Determined the output array with a given length
    Return:
        Zero numpy array for training neural decoding [num. electrodes x num. clusters x num. windows]
    """
    length_time_window = np.zeros((len(timestamps, )), dtype=np.uint32)
    num_clusters = np.zeros((len(timestamps, )), dtype=np.uint32)
    for idx, event_ch in enumerate(timestamps):
        length_time_window[idx] = 0 if len(event_ch) == 0 else event_ch[-1]
        num_clusters[idx] = 0 if len(event_ch) == 0 else np.unique(np.array(cluster[idx])).max() + 1

    if output_size == 0:
        num_windows = int(1 + np.ceil(length_time_window.max() / samples_time_window))
    else:
        num_windows = int(1 + np.ceil(output_size / samples_time_window))
    return np.zeros((len(timestamps), num_clusters.max() if use_cluster else 1, num_windows), dtype=np.uint16)


def __determine_firing_rate(timestamps: list, cluster: list, exp_sampling_rate_in_hertz, length_time_window_ms,
                            use_cluster=False, output_size=0) -> np.ndarray:
    """Pre-Processing Method: Calculating the firing rate for specific
    Args:
        timestamps: Lists with all timestamps of each electrode (iteration over electrode).
                    a timestamp is the no of the sample where the spike was detected
        cluster: Lists with all corresponding cluster unit of each timestamp
        samples_per_time_window: Size of the window for determining features
        use_cluster: Decision of cluster information will be used
        output_size: Determined the output array with a given length
    Return:
        Numpy array with windowed number of detected events for training neural decoding
        [num. electrodes x num. clusters x num. windows]
    """
    samples_per_time_window: int = int(1e-3 * exp_sampling_rate_in_hertz * length_time_window_ms)

    cluster_occurrency_per_timewindow = __generate_stream_empty_array(timestamps, cluster, samples_per_time_window, use_cluster, output_size)

    for electrode, timestamps_of_spiketiks_per_electrode in enumerate(timestamps):
        if len(timestamps_of_spiketiks_per_electrode) == 0:
            # Skip due to empty electrode events
            continue
        else:
            # "Slicing" the timestamps of choice electrode
            np_timestamps_of_spiketiks_per_electrode = np.array(timestamps_of_spiketiks_per_electrode)

            if use_cluster:
                all_cluster_of_selected_electrode = np.array(cluster[electrode])
                for cluster_num in np.unique(all_cluster_of_selected_electrode):
                    indices_of_selected_cluster_in_all_cluster_per_electrode = np.argwhere(
                        all_cluster_of_selected_electrode == cluster_num).flatten()

                    no_timewindows_of_selected_cluster_in_electrode = np.array(np.floor(np_timestamps_of_spiketiks_per_electrode[indices_of_selected_cluster_in_all_cluster_per_electrode] / samples_per_time_window),dtype=int)

                    time_windows_with_cluster_and_occurrences = np.unique(no_timewindows_of_selected_cluster_in_electrode, return_counts=True)
                    for i, time_window in enumerate(time_windows_with_cluster_and_occurrences[0]):
                        cluster_occurrency_per_timewindow[electrode, cluster_num, time_window] += time_windows_with_cluster_and_occurrences[1][i]
            else:
                no_timewindows_of_selected_cluster_in_electrode = np.array(
                    np.floor(np_timestamps_of_spiketiks_per_electrode / samples_per_time_window), dtype=int)
                time_windows_with_cluster_and_occurrences = np.unique(no_timewindows_of_selected_cluster_in_electrode, return_counts=True)
                for i, time_window in enumerate(time_windows_with_cluster_and_occurrences[0]):
                    cluster_occurrency_per_timewindow[electrode, 0, time_window] += time_windows_with_cluster_and_occurrences[1][i]
    return cluster_occurrency_per_timewindow


def __add_electrode_2d_mapping_to_dataset(data_raw, dataset_timestamps, dataset_waveform) -> list:
    """"""
    # exp_000 is enough because it´s the same for every experiment
    electrode_mapping = data_raw['exp_000']['orientation']
    dataset_timestamps0 = __translate_ts_datastream_into_picture(dataset_timestamps, electrode_mapping)
    # dataset_waveform0 = translate_wf_datastream_into_picture(dataset_waveform, electrode_mapping)
    return dataset_timestamps0


def __translate_ts_datastream_into_picture(data_raw: list, configuration: dict) -> list:
    """ Translate timestamp data stream into picture format """
    picture_data_raw = []
    picture_data_point = None
    labels = configuration['label']
    for data_point in data_raw:
        for electID, data in enumerate(data_point):
            if picture_data_point is None:
                picture_data_point = np.zeros((data.shape[0], 10, 10, data.shape[1]), dtype=np.uint16)

            for label in labels:
                if f"elec{electID + 1}" == label:  # elec bezieht sich auf den Datensatz nicht ändern!!
                    row = configuration['row'][95 - electID]
                    col = configuration['col'][95 - electID]
                    picture_data_point[:, col, row, :] = data

        picture_data_raw.append(picture_data_point)
    return picture_data_raw


def __translate_wf_datastream_into_picture(data_raw: list, configuration: dict) -> list:
    """ Translate waveform data stream into picture format"""
    picture_data_raw = []
    labels = configuration['label']
    picture_data_point = [[[] for _ in range(10)] for _ in range(10)]  # array of empty lists

    for data_point in data_raw:
        for electID, data in enumerate(data_point):
            for label in labels:
                if f"elec{electID + 1}" == label:  # electrodes from data start at 0, but in mapping at 1.
                    row = configuration['row'][
                        95 - electID]  # col, row range from 0-9 and point to the "correct" label from the mapping file
                    col = configuration['col'][95 - electID]
                    picture_data_point[col][row] = data  # set list of lists as our list at col,row

        picture_data_raw.append(picture_data_point)
        picture_data_point = [[[] for _ in range(10)] for _ in range(10)]

    return picture_data_raw


def __create_label_dict(dataset_decision) -> dict:
    """Creating the dictionary with labels"""
    label_dict = dict()
    num_label_types = 0
    for label in dataset_decision:
        used_label = label['patient_says']
        if isinstance(used_label, str):
            if used_label not in label_dict.keys():
                label_dict.update({used_label: num_label_types})
                num_label_types += 1
    return label_dict


def __counting_dataset(dataset_decision, label_dict):
    """"""
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
    num_samples = sum(label_count_label_free) + sum(label_count_label_made)
    return label_count_label_free, label_count_label_made, num_samples


if __name__ == "__main__":
    from package.dnn.pytorch_config_data import DefaultSettingsDataset

    default_settings = DefaultSettingsDataset
    default_settings.data_path = 'data'
    default_settings.data_file_name = '2024-02-05_Dataset-KlaesNeuralDecoding.npy'

    generate_electrode_mapping_from_data(default_settings, default_settings.get_path2folder_data)
