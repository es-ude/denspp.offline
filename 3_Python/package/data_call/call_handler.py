from os.path import join, exists, split
from glob import glob
import dataclasses
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
import csv
from package.structure_builder import create_folder_general_firstrun


@dataclasses.dataclass
class SettingsDATA:
    """Class for configuring the dataloader
    input:
    path        - Path to data storage
    data_set    - Type of dataset
    data_point  - Number within the dataset
    t_range     - List of the given time range for cutting the data [x, y]
    ch_sel      - List of electrodes to use [empty=all]
    fs_resample - Resampling frequency of the datapoint
    """
    path: str
    data_set: int
    data_case: int
    data_point: int
    t_range: list
    ch_sel: list
    fs_resample: float


RecommendedSettingsDATA = SettingsDATA(
    path="../2_Data",
    data_set=1, data_case=0, data_point=0,
    t_range=[0], ch_sel=[],
    fs_resample=100e3
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
        # --- Behaviour
        self.behaviour_exist = False
        self.behaviour = None

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


class _DataController:
    """Class for loading and manipulating the used dataset"""
    raw_data: DataHandler
    settings: SettingsDATA
    path2file: str
    path2mapping = ''
    csvfile_available = False

    def __init__(self) -> None:
        create_folder_general_firstrun()
        # --- Meta-Information about datasets
        self._no_subfolder = 0
        self._no_files = 0
        self.__fill_factor = 1
        self.__scaling = 1

    def do_cut(self) -> None:
        """Cutting all transient electrode signals in the given range"""
        t_range = np.array(self.settings.t_range)
        rawdata_in = self.raw_data.data_raw
        evnt_xpos_in = self.raw_data.evnt_xpos
        cluster_in = self.raw_data.evnt_id

        rawdata_out = list()
        evnt_xpos_out = list()
        cluster_out = list()

        if self.raw_data.data_fs_used == 0:
            self.raw_data.data_fs_used = self.raw_data.data_fs_orig

        # --- Positionen ermitteln
        if t_range.size == 2:
            idx0 = int(t_range[0] * self.raw_data.data_fs_used)
            idx1 = int(t_range[1] * self.raw_data.data_fs_used)
            self.__fill_factor = (idx0 - idx1) / rawdata_in[-1].size

            for idx, data_in in enumerate(rawdata_in):
                # --- Cutting specific time range out of raw data
                rawdata_out.append(data_in[idx0:idx1])

                # --- Cutting labeled information
                if self.raw_data.label_exist:
                    # Adapting new data
                    idx2 = int(np.argwhere(evnt_xpos_in[idx] >= idx0)[0])
                    idx3 = int(np.argwhere(evnt_xpos_in[idx] <= idx1)[-1])
                    evnt_xpos_out.append(evnt_xpos_in[idx][idx2:idx3] - idx0)
                    cluster_out.append(cluster_in[idx][idx2:idx3])

            # --- Return adapted data
            self.raw_data.data_raw = rawdata_out
            self.raw_data.evnt_xpos = evnt_xpos_out
            self.raw_data.evnt_id = cluster_out
            self.raw_data.data_time = float(rawdata_out[0].size / self.raw_data.data_fs_used)

    def do_resample(self) -> None:
        """Do resampling all transient signals"""
        desired_fs = self.settings.fs_resample
        do_resampling = bool(desired_fs != self.raw_data.data_fs_orig)

        data_out = list()
        spike_out = list()

        if do_resampling:
            self.raw_data.data_fs_used = desired_fs
            u_safe = 5e-6
            (p, q) = Fraction(self.raw_data.data_fs_used / self.raw_data.data_fs_orig).limit_denominator(10000).as_integer_ratio()
            self.__scaling = p / q

            for idx, data_in in enumerate(self.raw_data.data_raw):
                # --- Resampling the input
                u_chck = np.mean(data_in[0:10])
                if np.abs((u_chck < u_safe) - 1) == 1:
                    du = u_chck
                else:
                    du = 0

                data_out.append(du + resample_poly(data_in - du, p, q))

                # --- Resampling the labeled information
                if self.raw_data.label_exist:
                    spikepos_in = self.raw_data.evnt_xpos[idx]
                    spike_out.append(np.array(self.__scaling * spikepos_in, dtype=int))

            self.raw_data.data_raw = data_out
            self.raw_data.evnt_xpos = spike_out
        else:
            self.raw_data.data_fs_used = self.raw_data.data_fs_orig
            self.__scaling = 1

    def output_meta(self) -> None:
        """Print some meta information into the console"""
        print(f"... using data set of: {self.raw_data.data_name}"
              "\n... using data point:", self.path2file)
        if not self.raw_data.data_fs_used == 0 and not self.raw_data.data_fs_used == self.raw_data.data_fs_orig:
            fs_addon = f" (resampling to {int(1e-3 * self.raw_data.data_fs_used)} kHz)"
        else:
            fs_addon = ""
        print(f"... original sampling rate of {int(1e-3 * self.raw_data.data_fs_orig)} kHz{fs_addon}"
              f"\n... using {self.__fill_factor * 100:.2f}% of the data "
              f"(time length of {self.raw_data.data_time / self.__fill_factor:.2f} s)")

        if self.raw_data.label_exist:
            cluster_array = None
            # Extract number of cluster size in all inputs
            for idx, clid in enumerate(self.raw_data.evnt_id):
                if idx == 0:
                    cluster_array = clid
                else:
                    cluster_array = np.append(cluster_array, clid)
            cluster_no = np.unique(cluster_array)

            # Extract number of spikes in all inputs
            num_spikes = 0
            for idx, spk_num in enumerate(self.raw_data.evnt_xpos):
                num_spikes += spk_num.size

            print(f"... includes labels (noSpikes: {num_spikes} - noCluster: {cluster_no.size})")
        else:
            print(f"... has no labels / groundtruth")

    def get_data(self) -> DataHandler:
        """Calling the raw data with groundtruth of the called data"""
        self._transform_rawdata_to_numpy()
        return self.raw_data
        
    def generate_xpos_label(self, used_channel: int) -> np.ndarray:
        """Generating label ticks"""
        fs_used = self.raw_data.data_fs_used
        fs_orig = self.raw_data.data_fs_orig
        xpos_in = self.raw_data.evnt_xpos[used_channel]
        return xpos_in / fs_orig * fs_used
        
    def generate_label_stream_channel(self, used_channel: int, window_time=1.6e-3) -> np.ndarray:
        """Generating a transient array with labeling event detection
        Args:
            used_channel:   Number of used channel for labeling event detection
            window_time:    Time window of the trigger signal for generating the transient trigger array
        Returns:
            Numpy array with transient trigger signal
        """
        window_size = int(window_time * self.raw_data.data_fs_used)
        trgg0 = np.zeros(self.raw_data.data_raw[used_channel], dtype=int)
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
        for ch_used, trgg_used in enumerate(self.raw_data.evnt_xpos):
            trgg_out.append(self.generate_label_stream_channel(ch_used, window_time))
        return trgg_out

    def _prepare_call(self) -> None:
        """Loading the dataset"""
        # --- Checking if path is available
        if not exists(self.settings.path):
            raise FileNotFoundError(f"... data path of dataset not found! Please check")

    def _prepare_access_file(self, folder_name: str, data_type: str) -> None:
        """Getting the file of the corresponding trial"""
        sel_datacase = self.settings.data_case
        sel_datapoint = self.settings.data_point

        # --- Finding the right folder in data storage
        folder_structure = glob(join(self.settings.path, '*'))
        path2folder = ""
        for folder in folder_structure:
            if folder_name in folder:
                path2folder = folder

        # --- Checking for subfolder and file
        if path2folder:
            folder_structure = glob(join(path2folder, '*'))
            folder_content = glob(join(path2folder, data_type))

            if len(folder_content) == 0:
                # --- Taking datacase into account
                folder_content = glob(join(path2folder, folder_structure[sel_datacase], data_type))
                folder_content.sort()

            # --- Taking file
            try:
                self.path2file = folder_content[sel_datapoint]
                self._no_files = len(folder_content)
            except:
                if len(folder_content) < sel_datapoint:
                    raise ValueError("--- Variable sel_datapoint is higher then number of samples in dataset! ---")
                else:
                    raise FileNotFoundError("--- Files not available. Please check file structure! ---")
        else:
            raise FileNotFoundError("--- Folder not available - Please check folder name! ---")

    def _read_csv_file(self, path2csv: str, num_channels: int, split_option: str, start_pos_csvfile=0) -> list:
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

    def _transform_rawdata_from_csv_to_numpy(self, data: list) -> np.ndarray:
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
        if isinstance(self.raw_data.data_raw, list):
            num_channels = len(self.raw_data.data_raw)
            num_samples = np.zeros((num_channels, ), dtype=int)
            for idx, data in enumerate(self.raw_data.data_raw):
                num_samples[idx] = data.shape[0]

            data_out = np.zeros((num_channels, num_samples.min()), dtype=float)
            for idx, data in enumerate(self.raw_data.data_raw):
                data_out[idx, :] = data[0:num_samples.min()]

            self.raw_data.data_raw = data_out

    def do_mapping(self, path2csv="") -> None:
        """Transforming the input data to electrode array specific design
        (considering electrode format and coordination)
        Args:
            path2csv:   Path to csv file with information about electrode mapping (Default: "")
        Returns:
            None
        """
        # --- Checking if mapping file is available
        if path2csv == "":
            found_mapping_files = glob(join(split(self.path2file)[0], '*.csv'))
            self.path2mapping = found_mapping_files[0]
        else:
            self.path2mapping = path2csv

        # --- Generating mapping information
        self._generate_electrode_mapping_from_csv()
        self._generate_electrode_activation_mapping()
        self._transform_rawdata_mapping()

    def _generate_electrode_mapping_from_csv(self) -> None:
        """Function for reading the CSV file for electrode mapping of used (non-electrodes = 0)"""
        mapping_from_csv = []
        path2csv = self.path2mapping
        # --- Reading CSV file with electrode infos
        if not exists(path2csv):
            raise FileNotFoundError("CSV file for reading not found - Please check!")
        else:
            with open(path2csv, 'r') as file:
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

        electrode_mapping = self.raw_data.generate_empty_mapping_array_integer
        if not num_rows == self.raw_data.mapping_dimension[0] and not num_cols == self.raw_data.mapping_dimension[1]:
            raise ValueError("Array dimenson given in DataLoader and from csv file is not identical - Please check!")
        else:
            for row_idx, elec_row in enumerate(mapping_from_csv):
                for col_idx, elec_id in enumerate(elec_row):
                    if row_idx < electrode_mapping.shape[0] and col_idx < electrode_mapping.shape[1]:
                        electrode_mapping[row_idx, col_idx] = elec_id

            self.raw_data.mapping_exist = True
            self.raw_data.mapping_used = electrode_mapping

    def _generate_electrode_activation_mapping(self) -> None:
        """Generating the electrode activation map (Reference/Empty = False, Activity/Data = True)"""
        if not self.raw_data.mapping_exist:
            print("... skipped generation of electrode activitaion map")
        else:
            activation_map = self.raw_data.generate_empty_mapping_array_boolean
            posx, posy = np.where(self.raw_data.mapping_used != 0)
            for idx, pos_row in enumerate(posx):
                activation_map[pos_row, posy[idx]] = True
            self.raw_data.mapping_active = activation_map

    def _transform_rawdata_mapping(self) -> None:
        """Transforming the numpy array input to 2D array with electrode mapping configuration"""
        data_in = self.raw_data.data_raw
        data_map = self.raw_data.mapping_used

        if not self.raw_data.mapping_exist:
            print("... raw data array cannot be transformed into 2D-format")
            data_out = data_in
        else:
            data_out = np.zeros((data_map.shape[0], data_map.shape[1], data_in[0].size), dtype=float)
            for x in range(0, data_map.shape[0]):
                for y in range(0, data_map.shape[1]):
                    if self.raw_data.mapping_active[x, y]:
                        column = 0
                        # Searching the right index of electrode id to map id
                        for channel in self.raw_data.electrode_id:
                            if channel == data_map[x, y]:
                                data_out[x, y, :] = data_in[column]
                                break
                            column += 1
            print("... transforming raw data array from 1D to 2D")

        self.raw_data.data_raw = data_out


###########################################################################
if __name__ == "__main__":
    from package.data_call.call_spike_files import DataLoader, SettingsDATA
    from package.plot.plot_mea import results_mea_transient_total

    settings = SettingsDATA(
        path="C:\HomeOffice\Data_Neurosignal",
        data_set=8, data_case=1, data_point=1,
        t_range=[0, 0.5], ch_sel=[], fs_resample=20e3
    )
    data_loader = DataLoader(settings)
    data_loader.do_call()
    data_loader.do_cut()
    # data_loader.do_resample()
    data_loader.do_mapping()
    data = data_loader.get_data()

    results_mea_transient_total(data.data_raw, data, '../../runs/test', do_global_limit=True)
    results_mea_transient_total(data.data_raw, data, '../../runs/test', do_global_limit=False)
    del data_loader
    print(data)
