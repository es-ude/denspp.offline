import csv
import numpy as np
from dataclasses import dataclass
from os.path import join, exists, dirname, basename
from os import makedirs
from glob import glob
from fractions import Fraction
from scipy.signal import resample_poly
from denspp.offline.structure_builder import init_project_folder, get_path_project_start
from denspp.offline.data_call.owncloud_handler import OwncloudDownloader


@dataclass
class SettingsDATA:
    """Class for configuring the dataloader
    input:
    path        - Path to data storage
    data_set    - String with key for used data set
    data_point  - Number within the dataset
    t_range     - List of the given time range for cutting the data [x, y]
    ch_sel      - List of electrodes to use [empty=all]
    fs_resample - Resampling frequency of the datapoint
    do_mapping  - Decision if mapping (if available) is used
    """
    path: str
    data_set: str
    data_case: int
    data_point: int
    t_range: list
    ch_sel: list
    fs_resample: float
    do_mapping: bool


RecommendedSettingsDATA = SettingsDATA(
    path='data',
    data_set='quiroga',
    data_case=0, data_point=0,
    t_range=[0], ch_sel=[],
    fs_resample=100e3,
    do_mapping=True
)


class DataHandler:
    """Class with data and meta information of the used neural dataset"""
    def __init__(self):
        # --- Meta Information
        self.data_name = ''
        self.data_type = ''
        self.data_fs_orig = 0
        self.data_fs_used = 0
        self.data_time = 0.0
        # --- Raw data
        self.device_id = ''
        self.electrode_id = list()
        self.data_raw = list()
        # --- Electrode Design Information
        self.mapping_exist = False
        self.mapping_dimension = [1, 1]  # [row, colomn]
        self.mapping_used = self.generate_empty_mapping_array_integer
        self.mapping_active = self.generate_empty_mapping_array_boolean
        # --- GroundTruth fpr Event Signal Processing
        self.label_exist = False
        self.evnt_xpos = list()
        self.evnt_id = list()

    @property
    def generate_empty_mapping_array_integer(self) -> np.ndarray:
        return np.zeros((self.mapping_dimension[0], self.mapping_dimension[1]), dtype=int)

    @property
    def generate_empty_mapping_array_boolean(self) -> np.ndarray:
        return np.zeros((self.mapping_dimension[0], self.mapping_dimension[1]), dtype=bool)


def translate_unit_to_scale_value(unit_str: str, pos: int) -> float:
    """Translating the unit of a value from string to float"""
    gain_base = 1e0
    if not len(unit_str) == 1:
        if unit_str[pos] == 'm':
            gain_base = 1e-3
        elif unit_str[pos] == 'u':
            gain_base = 1e-6
        elif unit_str[pos] == 'n':
            gain_base = 1e-9
    return gain_base


def transform_label_from_csv_to_numpy(marker_loaded: list, label_text: list, start_pos: int) -> [list, list]:
    """Translating the event labels from csv file to numpy
    Args:
        marker_loaded:
        label_text:
        start_pos:      Start position from reading the csv file
    Returns:
        Two list with type information and marker information
        """
    loaded_type = list()
    for idx, label in enumerate(marker_loaded[0]):
        if idx > start_pos:
            id = -1
            for num, type in enumerate(label_text):
                if type in label:
                    id = num
            loaded_type.append(id)

    loaded_marker = list()
    for idx, label in enumerate(marker_loaded[1]):
        if idx > 0:
            loaded_marker.append(int(label[:-1]))

    return loaded_type, loaded_marker


class DataController:
    """Class for loading and manipulating the used dataset"""
    __download_handler = OwncloudDownloader(use_dataset=False)
    _raw_data: DataHandler
    _settings: SettingsDATA
    _path2file: str = ''
    _path2folder: str = ''
    path2mapping_local: list = []
    path2mapping_remote: list = []
    path2mapping: str = ''

    def __init__(self) -> None:
        init_project_folder()
        self.__fill_factor = 1
        self.__scaling = 1
        self._methods_available = dir(DataController)
        self.__default_data_path = join(get_path_project_start(), 'data')
        self.__config_data_selection = [self.__default_data_path, 0, 0]

    def do_cut(self) -> None:
        """Cutting all transient electrode signals in the given range"""
        t_range = np.array(self._settings.t_range)
        rawdata_in = self._raw_data.data_raw
        evnt_xpos_in = self._raw_data.evnt_xpos
        cluster_in = self._raw_data.evnt_id

        rawdata_out = list()
        evnt_xpos_out = list()
        cluster_out = list()

        if self._raw_data.data_fs_used == 0:
            self._raw_data.data_fs_used = self._raw_data.data_fs_orig

        # --- Getting the positition of used time range
        if t_range.size == 2:
            idx0 = int(t_range[0] * self._raw_data.data_fs_used)
            idx1 = int(t_range[1] * self._raw_data.data_fs_used)
            self.__fill_factor = (idx0 - idx1) / rawdata_in[-1].size

            for idx, data_in in enumerate(rawdata_in):
                # --- Cutting specific time range out of raw data
                rawdata_out.append(data_in[idx0:idx1])

                # --- Cutting labeled information
                if self._raw_data.label_exist:
                    # Adapting new data
                    pos_start = np.argwhere(evnt_xpos_in[idx] >= idx0).flatten()
                    idx2 = int(pos_start[0]) if pos_start.size > 0 else -1
                    pos_stopp = np.argwhere(evnt_xpos_in[idx] <= idx1).flatten()
                    idx3 = int(pos_stopp[-1]) if pos_stopp.size > 0 else -1

                    if idx2 == -1 or idx3 == -1:
                        evnt_xpos_out.append([])
                        cluster_out.append([])
                    else:
                        evnt_xpos_out.append(evnt_xpos_in[idx][idx2:idx3] - idx0)
                        cluster_out.append(cluster_in[idx][idx2:idx3])

            # --- Return adapted data
            self._raw_data.data_raw = rawdata_out
            self._raw_data.evnt_xpos = evnt_xpos_out
            self._raw_data.evnt_id = cluster_out
            self._raw_data.data_time = float(rawdata_out[0].size / self._raw_data.data_fs_used)

    def do_resample(self) -> None:
        """Do resampling all transient signals"""
        desired_fs = self._settings.fs_resample
        do_resampling = bool(desired_fs != self._raw_data.data_fs_orig)

        data_out = list()
        spike_out = list()

        if do_resampling:
            self._raw_data.data_fs_used = desired_fs
            u_safe = 5e-6
            (p, q) = Fraction(self._raw_data.data_fs_used / self._raw_data.data_fs_orig).limit_denominator(10000).as_integer_ratio()
            self.__scaling = p / q

            for idx, data_in in enumerate(self._raw_data.data_raw):
                # --- Resampling the input
                u_chck = np.mean(data_in[0:10])
                if np.abs((u_chck < u_safe) - 1) == 1:
                    du = u_chck
                else:
                    du = 0
                data_out.append(du + resample_poly(data_in - du, p, q))

                # --- Resampling the labeled information
                if self._raw_data.label_exist:
                    spikepos_in = self._raw_data.evnt_xpos[idx]
                    spike_out.append(np.array(self.__scaling * spikepos_in, dtype=int))

            self._raw_data.data_raw = data_out
            self._raw_data.evnt_xpos = spike_out
        else:
            self._raw_data.data_fs_used = self._raw_data.data_fs_orig
            self.__scaling = 1

    def output_meta(self) -> None:
        """Print some meta information into the console"""
        print(f"... using data set of: {self._raw_data.data_name}"
              "\n... using data point:", self._path2file)
        if not self._raw_data.data_fs_used == 0 and not self._raw_data.data_fs_used == self._raw_data.data_fs_orig:
            fs_addon = f" (resampling to {int(1e-3 * self._raw_data.data_fs_used)} kHz)"
        else:
            fs_addon = ""
        print(f"... original sampling rate of {int(1e-3 * self._raw_data.data_fs_orig)} kHz{fs_addon}"
              f"\n... using {self.__fill_factor * 100:.2f}% of the data "
              f"(time length of {self._raw_data.data_time / self.__fill_factor:.2f} s)")

        if self._raw_data.label_exist:
            cluster_array = None
            # Extract number of cluster size in all inputs
            for idx, clid in enumerate(self._raw_data.evnt_id):
                if idx == 0:
                    cluster_array = clid
                else:
                    cluster_array = np.append(cluster_array, clid)
            cluster_no = np.unique(cluster_array)

            # Extract number of spikes in all inputs
            num_spikes = 0
            for idx, spk_num in enumerate(self._raw_data.evnt_xpos):
                num_spikes += spk_num.size

            print(f"... includes labels (noSpikes: {num_spikes} - noCluster: {cluster_no.size})")
        else:
            print(f"... has no labels / groundtruth")

    def get_data(self) -> DataHandler:
        """Calling the raw data with groundtruth of the called data"""
        self._transform_rawdata_to_numpy()
        return self._raw_data
        
    def generate_xpos_label(self, used_channel: int) -> np.ndarray:
        """Generating label ticks"""
        fs_used = self._raw_data.data_fs_used
        fs_orig = self._raw_data.data_fs_orig
        xpos_in = self._raw_data.evnt_xpos[used_channel]
        return xpos_in / fs_orig * fs_used
        
    def generate_label_stream_channel(self, used_channel: int, window_time=1.6e-3) -> np.ndarray:
        """Generating a transient array with labeling event detection
        Args:
            used_channel:   Number of used channel for labeling event detection
            window_time:    Time window of the trigger signal for generating the transient trigger array
        Returns:
            Numpy array with transient trigger signal
        """
        window_size = int(window_time * self._raw_data.data_fs_used)
        trgg0 = np.zeros(self._raw_data.data_raw[used_channel], dtype=int)
        for val in self.generate_xpos_label(used_channel):
            trgg0[int(val):int(val) + window_size] = 1
        return trgg0

    def generate_label_stream_all(self, window_time=1.6e-3) -> list:
        """Generating a list with transient arrays to label event detection of all used channels
        Args:
            window_time:    Time window of the trigger signal for generating the transient trigger array
        Returns:
            List with numpy array of transient trigger signal
        """
        trgg_out = list()
        for ch_used, trgg_used in enumerate(self._raw_data.evnt_xpos):
            trgg_out.append(self.generate_label_stream_channel(ch_used, window_time))
        return trgg_out

    def __get_data_available_local(self, folder_name: str, data_type: str, path_ref: str = '') -> str:
        folder_structure = glob(join(self.__config_data_selection[0], '*'))
        path2folder = [folder for folder in folder_structure if folder_name in folder]
        file_name = basename(path_ref)

        if len(path2folder) == 0:
            return ""
        else:
            self._path2folder = path2folder[0]
            self.path2mapping_local = glob(join(path2folder[0], 'Mapping_*.csv'))
            folder_structure = glob(join(path2folder[0], '*'))
            folder_content = glob(join(path2folder[0], data_type))
            if len(folder_structure) > len(folder_content):
                folder_content = glob(join(folder_structure[self.__config_data_selection[1]], data_type))
                folder_content.sort()
            elif len(folder_content):
                folder_content.sort()

            chck = [file for file in folder_content if file_name in file]
            return "" if len(chck) == 0 else chck[0]

    def __get_path_available_remote(self, folder_name: str, data_type: str) -> str:
        overview = self.__download_handler.get_overview_folder()
        path2folder = [s for s in overview if any(folder_name in s for xs in overview)]
        self.path2mapping_remote = self.__download_handler.get_overview_data(path2folder[0], 'Mapping_*.csv')

        if len(path2folder) == 0:
            return ""
        else:
            folder_structure = self.__download_handler.get_overview_folder(path2folder[0])
            if len(folder_structure):
                folder_content = self.__download_handler.get_overview_data(folder_structure[self.__config_data_selection[1]], data_type)
            else:
                folder_content = self.__download_handler.get_overview_data(path2folder[0], data_type)
            folder_content.sort()
            return folder_content[self.__config_data_selection[2]]

    def _prepare_access_file(self, folder_name: str, data_type: str) -> None:
        """Getting the file of the corresponding trial"""
        used_datapath = self.__default_data_path if self._settings.path == '' else self._settings.path
        self.__config_data_selection = [used_datapath, self._settings.data_case, self._settings.data_point]

        path2chck = self.__get_path_available_remote(folder_name, data_type)
        pathlocal = self.__get_data_available_local(folder_name, data_type, path2chck)
        if basename(pathlocal) == basename(path2chck) and path2chck:
            self._path2file = pathlocal
        elif path2chck and not pathlocal:
            path2data = join(self._settings.path, dirname(path2chck[1:]))
            path2file = join(self._settings.path, path2chck[1:])
            makedirs(path2data, exist_ok=True)
            self.__download_handler.download_file(path2chck, path2file)
            self._path2file = path2file
        else:
            raise FileNotFoundError("--- File is not available. Please check! ---")

    @staticmethod
    def _read_csv_file(path2csv: str, num_channels: int, split_option: str, start_pos_csvfile=0) -> list:
        """Reading the csv file
        Args:
            path2csv:           Path to csv file for reading content
            num_channels:       Given number of channels for seperating the list
            split_option:       Option for splitting the strings of csv reading
            start_pos_csvfile:  Selection list element from csv line-read for processing
        Returns:
            One list with converted informations
        """
        if not num_channels == 1:
            loaded_data = [[] for idx in range(num_channels)]
        else:
            loaded_data = []

        if not exists(path2csv):
            print("... file not available. Electrode mapping will be skipped")
            return []
        else:
            file = open(path2csv, 'r')
            for line in file:
                input = line.split(split_option)
                sel_list = start_pos_csvfile
                for val in input:
                    if val:
                        loaded_data[sel_list].append(val)
                        sel_list += 1
            return loaded_data

    @staticmethod
    def _transform_rawdata_from_csv_to_numpy(data: list) -> np.ndarray:
        """Tranforming the csv data to numpy array"""
        # --- Getting meta information
        num_samples = list()
        for idx, data0 in enumerate(data):
            num_samples.append(len(data0))
        num_samples = np.array(num_samples)
        num_channels = len(data) + 1

        # --- Getting data in right format
        data_used = np.zeros((num_channels, num_samples.min()), dtype=float)
        for idx, data_ch in enumerate(data):
            data_ch0 = list()
            for value in data_ch:
                data_ch0.append(float(value))
            data_used[idx, :] = np.array(data_ch0[0:num_samples.min()])

        return data_used

    def _transform_rawdata_to_numpy(self) -> None:
        """Transforming the initial raw data from list to numpy array"""
        if isinstance(self._raw_data.data_raw, list):
            num_channels = len(self._raw_data.data_raw)
            num_samples = np.zeros((num_channels, ), dtype=int)
            for idx, data in enumerate(self._raw_data.data_raw):
                num_samples[idx] = data.shape[0]

            data_out = np.zeros((num_channels, num_samples.min()), dtype=float)
            for idx, data in enumerate(self._raw_data.data_raw):
                data_out[idx, :] = data[0:num_samples.min()]

            self._raw_data.data_raw = data_out

    def do_mapping(self, path2csv="") -> None:
        """Transforming the input data to electrode array specific design
        (considering electrode format and coordination)
        Args:
            path2csv:   Path to csv file with information about electrode mapping (Default: "")
        Returns:
            None
        """
        # --- Checking if mapping file is available
        if not path2csv:
            if len(self.path2mapping_local) == 0 and len(self.path2mapping_remote):
                self.path2mapping = join(self._path2folder, basename(self.path2mapping_remote[0]))
                self.__download_handler.download_file(self.path2mapping_remote[0], self.path2mapping)
            elif len(self.path2mapping_local) == 0 and len(self.path2mapping_remote) == 0:
                self.path2mapping = ''
            else:
                self.path2mapping = self.path2mapping_local[0]
        else:
            self.path2mapping = path2csv

        # --- Generating mapping information
        if self._settings.do_mapping and exists(self.path2mapping) and self.path2mapping:
            self._generate_electrode_mapping_from_csv()
            self._generate_electrode_activation_mapping()
            self._transform_rawdata_mapping()

    def _generate_electrode_mapping_from_csv(self) -> None:
        """Function for reading the CSV file for electrode mapping of used (non-electrodes = 0)"""
        mapping_from_csv = []
        # --- Reading CSV file with electrode infos
        with open(self.path2mapping, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                row = row[0]
                sep_row = row.split(';')
                numbers_row = [int(value) for value in sep_row]
                if numbers_row:
                    mapping_from_csv.append(numbers_row)

        # --- Generating numpy array
        num_rows = len(mapping_from_csv)
        num_cols = len(mapping_from_csv[0])

        electrode_mapping = self._raw_data.generate_empty_mapping_array_integer
        if not num_rows == self._raw_data.mapping_dimension[0] and not num_cols == self._raw_data.mapping_dimension[1]:
            raise ValueError("Array dimenson given in DataLoader and from csv file is not identical - Please check!")
        else:
            for row_idx, elec_row in enumerate(mapping_from_csv):
                for col_idx, elec_id in enumerate(elec_row):
                    if row_idx < electrode_mapping.shape[0] and col_idx < electrode_mapping.shape[1]:
                        electrode_mapping[row_idx, col_idx] = elec_id

            self._raw_data.mapping_exist = True
            self._raw_data.mapping_used = electrode_mapping

    def _generate_electrode_activation_mapping(self) -> None:
        """Generating the electrode activation map (Reference/Empty = False, Activity/Data = True)"""
        if not self._raw_data.mapping_exist:
            print("... skipped generation of electrode activitaion map")
        else:
            activation_map = self._raw_data.generate_empty_mapping_array_boolean
            posx, posy = np.where(self._raw_data.mapping_used != 0)
            for idx, pos_row in enumerate(posx):
                activation_map[pos_row, posy[idx]] = True
            self._raw_data.mapping_active = activation_map

    def _transform_rawdata_mapping(self) -> None:
        """Transforming the numpy array input to 2D array with electrode mapping configuration"""
        data_in = self._raw_data.data_raw
        data_map = self._raw_data.mapping_used

        if not self._raw_data.mapping_exist:
            print("... raw data array cannot be transformed into 2D-format")
            data_out = data_in
        else:
            data_out = np.zeros((data_map.shape[0], data_map.shape[1], data_in[0].size), dtype=float)
            for x in range(0, data_map.shape[0]):
                for y in range(0, data_map.shape[1]):
                    if self._raw_data.mapping_active[x, y]:
                        column = 0
                        # Searching the right index of electrode id to map id
                        for channel in self._raw_data.electrode_id:
                            if channel == data_map[x, y]:
                                data_out[x, y, :] = data_in[column]
                                break
                            column += 1
            print("... transforming raw data array from 1D to 2D")
        self._raw_data.data_raw = data_out

    def do_call(self):
        """Loading the dataset"""
        # --- Searching the load function for dataset translation
        methods_list_all = [method for method in self._methods_available]
        search_param = '_DataLoader'
        methods_load_data = [method for method in methods_list_all if search_param in method]

        # --- Getting the function to call
        used_data_source_idx = -1
        warning_text = "\nPlease select key words in variable 'data_set' for calling methods to read transient data"
        warning_text += "\n=========================================================================================="
        for method in methods_load_data:
            warning_text += f"\n\t{method}"

        for idx, method in enumerate(methods_load_data):
            if self._settings.data_set in method:
                used_data_source_idx = idx
                break

        # --- Call the function
        if not self._settings.data_set or used_data_source_idx == -1:
            raise ValueError(warning_text)
        else:
            getattr(self, methods_load_data[idx])()
