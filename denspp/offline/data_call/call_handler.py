import numpy as np
from logging import getLogger, Logger
from dataclasses import dataclass
from os.path import join, exists, dirname, basename
from os import makedirs
from glob import glob
from pathlib import Path
from fractions import Fraction
from scipy.signal import resample_poly
from denspp.offline import get_path_to_project_start
from denspp.offline.csv_handler import CsvHandler
from denspp.offline.data_call.owncloud_handler import OwnCloudDownloader


@dataclass
class SettingsData:
    """Class for configuring the dataloader
    Attributes:
    path:           Path to data storage
    data_set:       String with key for used data set
    data_point:     Number within the dataset
    t_range:        List of the given time range for cutting the data [x, y]
    ch_sel:         List of electrodes to use [empty=all]
    fs_resample:    Resampling frequency of the datapoint (== 0.0, no resampling)
    do_mapping:     Decision if mapping (if available) is used
    mapping_type:   Object type for loading mapping file (int or str)
    """
    path: str
    data_set: str
    data_case: int
    data_point: int
    t_range: list
    ch_sel: list
    fs_resample: float
    do_mapping: bool
    mapping_type: object


DefaultSettingsData = SettingsData(
    path='data',
    data_set='quiroga',
    data_case=0, data_point=0,
    t_range=[0], ch_sel=[],
    fs_resample=100e3,
    do_mapping=True,
    mapping_type=int
)


class DataHandler:
    """Class with data and meta information of the used neural dataset"""
    def __init__(self):
        # --- Meta Information
        self.data_name: str = ''
        self.data_type: str = ''
        self.data_fs_orig: float = 0.
        self.data_fs_used: float = 0.
        self.data_time: float = 0.0
        # --- Raw data
        self.electrode_id: list = list()
        self.data_raw: np.ndarray
        # --- Electrode Design Information
        self.mapping_exist: bool=False
        self.mapping_dimension: list = [1, 1]  # [row, colomn]
        self.mapping_used: np.ndarray = self.generate_empty_mapping_array_integer
        self.mapping_active: np.ndarray = self.generate_empty_mapping_array_boolean
        # --- GroundTruth fpr Event Signal Processing
        self.label_exist: bool=False
        self.evnt_xpos: np.ndarray = np.zeros((1, ), dtype=int)
        self.evnt_id: np.ndarray = np.zeros((1, ), dtype=int)

    @property
    def generate_empty_mapping_array_integer(self) -> np.ndarray:
        return np.zeros((self.mapping_dimension[0], self.mapping_dimension[1]), dtype=int)

    @property
    def generate_empty_mapping_array_boolean(self) -> np.ndarray:
        return np.zeros((self.mapping_dimension[0], self.mapping_dimension[1]), dtype=bool)


class ControllerData:
    __logger: Logger
    __download_handler: OwnCloudDownloader
    _raw_data: DataHandler
    _settings: SettingsData
    _path2file: str=''
    _path2folder_remote: str=''
    _path2folder_local: str=''
    _path2mapping: str=''

    def __init__(self) -> None:
        """Class for loading and manipulating the used dataset"""
        self.__logger = getLogger(__name__)
        self.__fill_factor = 1
        self.__scaling = 1
        self._methods_available = dir(ControllerData)
        self.__default_data_path = join(get_path_to_project_start(), 'data')
        self.__config_data_selection = [self.__default_data_path, 0, 0]
        self.__download_handler = OwnCloudDownloader()

    @staticmethod
    def _extract_func(class_obj: object) -> list:
        return  [method for method in dir(class_obj) if class_obj.__name__ in method]

    def do_cut(self) -> None:
        """Cutting all transient electrode signals in the given range"""
        if self._raw_data.data_fs_used == 0:
            self._raw_data.data_fs_used = self._raw_data.data_fs_orig

        # --- Getting the positition of used time range
        t_range = np.array(self._settings.t_range)
        if t_range.size == 2:
            rawdata_in = self._raw_data.data_raw
            evnt_xpos_in = self._raw_data.evnt_xpos
            cluster_in = self._raw_data.evnt_id

            idx0 = int(t_range[0] * self._raw_data.data_fs_used)
            idx1 = int(t_range[1] * self._raw_data.data_fs_used)

            # --- Cutting labeled information
            if self._raw_data.label_exist:
                # Adapting new data
                pos_start = np.argwhere(evnt_xpos_in >= idx0).flatten()
                idx2 = int(pos_start[0]) if pos_start.size > 0 else -1
                pos_stopp = np.argwhere(evnt_xpos_in <= idx1).flatten()
                idx3 = int(pos_stopp[-1]) if pos_stopp.size > 0 else -1

                if idx2 == -1 or idx3 == -1:
                    evnt_xpos_out = list()
                    cluster_out = list()
                else:
                    evnt_xpos_out = evnt_xpos_in[idx2:idx3] - idx0
                    cluster_out = cluster_in[idx2:idx3]
                self._raw_data.evnt_xpos = evnt_xpos_out
                self._raw_data.evnt_id = cluster_out

            # --- Return adapted data
            self._raw_data.data_raw = rawdata_in[:, idx0:idx1] if len(rawdata_in.shape) == 2 else rawdata_in[idx0:idx1]
            self._raw_data.data_time = float(self._raw_data.data_raw.shape[-1] / self._raw_data.data_fs_used)
            self.__fill_factor = self._raw_data.data_raw.shape[-1] / rawdata_in.shape[-1]

    def do_resample(self) -> None:
        """Do resampling all transient signals"""
        desired_fs = self._settings.fs_resample
        do_resampling = bool(desired_fs != self._raw_data.data_fs_orig) and desired_fs != 0.0

        if do_resampling:
            self._raw_data.data_fs_used = desired_fs
            u_safe = 5e-6
            (p, q) = Fraction(self._raw_data.data_fs_used / self._raw_data.data_fs_orig).limit_denominator(10000).as_integer_ratio()
            self.__scaling = p / q

            data_in = self._raw_data.data_raw
            # --- Resampling the input
            u_chck = np.mean(data_in[0:10])
            if np.abs((u_chck < u_safe) - 1) == 1:
                du = u_chck
            else:
                du = 0
            data_out = du + resample_poly(data_in - du, p, q)

            # --- Resampling the labeled information
            if self._raw_data.label_exist:
                spike_out = np.array(self.__scaling * self._raw_data.evnt_xpos, dtype=int)
            else:
                spike_out = list()

            self._raw_data.data_raw = data_out
            self._raw_data.evnt_xpos = spike_out
        else:
            self._raw_data.data_fs_used = self._raw_data.data_fs_orig
            self.__scaling = 1

    def output_meta(self) -> None:
        """Print some meta information into the console"""
        self.__logger.info(f"... using data set of: {self._raw_data.data_name}")
        self.__logger.info("... using data point:", self._path2file)
        if not self._raw_data.data_fs_used == 0 and not self._raw_data.data_fs_used == self._raw_data.data_fs_orig:
            fs_addon = f" (resampling to {int(1e-3 * self._raw_data.data_fs_used)} kHz)"
        else:
            fs_addon = ""
        self.__logger.info(f"... original sampling rate of {int(1e-3 * self._raw_data.data_fs_orig)} kHz{fs_addon}")
        self.__logger.info(f"... using {self.__fill_factor * 100:.2f}% of the data (time length of {self._raw_data.data_time / self.__fill_factor:.2f} s)")

        if self._raw_data.label_exist:
            cluster_array = None
            # Extract number of cluster size in all inputs
            for idx, clid in enumerate(self._raw_data.evnt_id):
                if idx == 0:
                    cluster_array = clid
                else:
                    cluster_array = np.append(cluster_array, clid)
            cluster_no = np.unique(cluster_array)
            self.__logger.info(f"... includes labels (noSpikes: {self._raw_data.evnt_xpos.size} - noCluster: {cluster_no.size})")
        else:
            self.__logger.info(f"... has no labels / groundtruth")

    def get_data(self) -> DataHandler:
        """Calling the raw data with groundtruth of the called data"""
        return self._raw_data

    def __get_data_available_local(self, path_ref: str, folder_name: str, data_type: str) -> str:
        """Function for getting the path to file from remote
        :param path_ref:    Part to the reference folder from remote
        :param folder_name: Part to the folder name to find local
        :param data_type:   String with the data type to find local
        :return:            File name
        """
        if path_ref:
            # --- If data is online available - Checking if is is also local available
            path2chck = join(self.__config_data_selection[0], dirname(path_ref).lstrip("/"), basename(path_ref))
            self._path2folder_local = '' if not exists(path2chck) else dirname(join(*[path_seg for path_seg in Path(path2chck).parts[:2]], ''))
            return '' if not exists(path2chck) else path2chck
        else:
            # --- Routine for find local data if online is not available
            folder2search = [folder for folder in glob(join(self.__config_data_selection[0], '*')) if folder_name in folder][0]
            if not folder2search:
                raise FileNotFoundError(f"Folder with index {folder2search} not found (locally and remotely)!")
            else:
                self._path2folder_local = folder2search
                folder_content = glob(join(folder2search, data_type))
                if len(folder_content) == 0:
                    # --- Go into next folder structure and look there
                    folder_structure = glob(join(folder2search, '*'))
                    folder2search = join(folder2search, self.__config_data_selection[1]) if type(
                        self.__config_data_selection[1]) == str else join(
                        folder_structure[self.__config_data_selection[1]])
                    folder_content = glob(join(folder2search, data_type))
                    folder_content.sort()
                else:
                    # --- Look in folder_content for file
                    pass

                # --- Getting the
                file_name = join(folder2search, self.__config_data_selection[2]) if type(self.__config_data_selection[2]) == str else join(folder_content[self.__config_data_selection[2]])
                if file_name:
                    return file_name
                else:
                    raise FileNotFoundError(f"File is not available (locally and remotely)!")

    def __get_path_available_remote(self, folder_name: str, data_type: str) -> str:
        """Function for getting the path to file from remote
        :param folder_name:     Part of the folder name to find remote
        :param data_type:       String with the data type to find remote
        :return:                File name
        """
        # --- Try to get the path from remote
        try:
            overview = self.__download_handler.get_overview_folder(False)
            path2folder = [s for s in overview if any(folder_name in s for xs in overview)]
        except:
            return ""

        if len(path2folder) == 0:
            return ""
        else:
            self._path2folder_remote = path2folder[0]
            folder_structure = self.__download_handler.get_overview_folder(False, path2folder[0])
            if len(folder_structure):
                folder_search = join(path2folder[0], self.__config_data_selection[1]) if type(self.__config_data_selection[1]) == str else folder_structure[self.__config_data_selection[1]]
                folder_content = self.__download_handler.get_overview_data(False, folder_search, data_type)
            else:
                folder_content = self.__download_handler.get_overview_data(False, path2folder[0], data_type)
            folder_content.sort()
            return folder_content[self.__config_data_selection[2]]

    def _prepare_access_file(self, folder_name: str, data_type: str) -> None:
        """Getting the file of the corresponding trial
        :param folder_name:     String with folder name where the data is located
        :param data_type:       String with data type of the data
        """
        used_datapath = self.__default_data_path if self._settings.path == '' else self._settings.path
        self.__config_data_selection = [used_datapath, self._settings.data_case, self._settings.data_point]

        path2remote = self.__get_path_available_remote(folder_name, data_type)
        path2local = self.__get_data_available_local(path2remote, folder_name, data_type)
        if path2local:
            self._path2file = path2local
        elif path2remote and not path2local:
            path2data = join(self._settings.path, dirname(path2remote[1:]))
            path2file = join(self._settings.path, path2remote[1:])
            makedirs(path2data, exist_ok=True)
            self.__download_handler.download_file(use_dataset=False, file_name=path2remote, destination_download=path2file)
            self._path2file = path2file
        else:
            raise FileNotFoundError("--- File is not available. Please check! ---")

    def do_mapping(self, path2csv: str='', index_search: str='Mapping_*.csv') -> None:
        """Transforming the input data to electrode array specific design
        (considering electrode format and coordination)
        :parm path2csv:     Path to csv file with information about electrode mapping (Default: "")
        :return:            None
        """
        if path2csv:
            self._path2mapping = path2csv
        else:
            # --- Checking if mapping file is available
            mapping_local = glob(join(self._path2folder_local, index_search)) if self._path2folder_local else []
            mapping_remote = self.__download_handler.get_overview_data(
                use_dataset=False,
                search_folder=self._path2folder_remote,
                format=index_search
            ) if self._path2folder_remote else []

            # --- Getting the file
            if len(mapping_local):
                self._path2mapping = mapping_local[0]
            elif len(mapping_remote):
                self._path2mapping = join(self._path2folder_local, basename(mapping_remote[0]))
                self.__download_handler.download_file(
                    use_dataset=False,
                    file_name=mapping_remote[0],
                    destination_download=self._path2mapping
                )
            else:
                self._path2mapping = ''

        # --- Generating mapping information
        if self._settings.do_mapping and exists(self._path2mapping):
            self.__logger.info("Apply electrode mapping")
            self._generate_electrode_mapping_from_csv()
            self._generate_electrode_activation_mapping()
            self._transform_rawdata_mapping()
        else:
            self.__logger.info("No electrode mapping is available")

    def _generate_electrode_mapping_from_csv(self) -> None:
        """Function for reading the CSV file for electrode mapping of used (non-electrodes = 0)"""
        mapping_from_csv = CsvHandler(
            path=dirname(self._path2mapping),
            file_name=basename(self._path2mapping),
            delimiter=';'
        ).read_data_from_csv(
            include_chapter_line=False,
            start_line=0,
            type_load=self._settings.mapping_type
        ).tolist()

        # --- Generating numpy array
        num_rows = len(mapping_from_csv)
        num_cols = len(mapping_from_csv[0])

        electrode_mapping = self._raw_data.generate_empty_mapping_array_integer
        if not num_rows == self._raw_data.mapping_dimension[0] and not num_cols == self._raw_data.mapping_dimension[1]:
            text = "Array dimenson given in DataLoader and from csv file is not identical - Please check!"
            self.__logger.error(text)
            raise ValueError(text)
        else:
            for row_idx, elec_row in enumerate(mapping_from_csv):
                for col_idx, elec_id in enumerate(elec_row):
                    if row_idx < electrode_mapping.shape[0] and col_idx < electrode_mapping.shape[1]:
                        if type(elec_id) == str:
                            val = [idx + 1 for idx, val in enumerate(self._raw_data.electrode_id) if val == elec_id]
                            electrode_mapping[row_idx, col_idx] = val[0] if len(val) else 0
                        else:
                            electrode_mapping[row_idx, col_idx] = elec_id

            self._raw_data.mapping_exist = True
            self._raw_data.mapping_used = electrode_mapping

    def _generate_electrode_activation_mapping(self) -> None:
        """Generating the electrode activation map (Reference/Empty = False, Activity/Data = True)"""
        if not self._raw_data.mapping_exist:
            self.__logger.info("... skipped generation of electrode activitaion map")
        else:
            activation_map = self._raw_data.generate_empty_mapping_array_boolean
            posx, posy = np.where(self._raw_data.mapping_used != 0)
            for idx, pos_row in enumerate(posx):
                activation_map[pos_row, posy[idx]] = True
            self._raw_data.mapping_active = activation_map

    def _transform_rawdata_mapping(self) -> None:
        """Transforming the numpy array input to 2D array with electrode mapping configuration"""
        if not self._raw_data.mapping_exist:
            self.__logger.info("... raw data array cannot be transformed into 2D-format")
        else:
            data_map = self._raw_data.mapping_used
            data_out = self._raw_data.data_raw
            dut = np.zeros((data_map.shape[0], data_map.shape[1], data_out.shape[-1]), dtype=data_out.dtype)

            for x in range(0, data_map.shape[0]):
                for y in range(0, data_map.shape[1]):
                    if self._raw_data.mapping_active[x, y]:
                        column = 0
                        # Searching the right index of electrode id to map id
                        for channel in self._raw_data.electrode_id:
                            if channel == data_map[x, y]:
                                dut[x, y, :] = data_out[column]
                                break
                            column += 1
            self.__logger.info("... transforming raw data array from 1D to 2D")
            self._raw_data.data_raw = dut

    def do_call(self):
        """Loading the dataset"""
        # --- Getting the function to call
        warning_text = "\nPlease select key words in variable 'data_set' for calling methods to read transient data"
        warning_text += "\n=========================================================================================="
        for method in self._methods_available:
            warning_text += f"\n\t{method}"

        # --- Call the function
        used_data_source_idx = -1
        for idx, method in enumerate(self._methods_available):
            if self._settings.data_set in method:
                used_data_source_idx = idx
                break

        if not self._settings.data_set or used_data_source_idx == -1:
            self.__logger.error(warning_text)
            raise ValueError
        else:
            getattr(self, self._methods_available[used_data_source_idx])()
