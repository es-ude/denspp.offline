import numpy as np
from logging import getLogger, Logger
from dataclasses import dataclass
from os.path import join, exists, dirname, basename
from os import makedirs
from glob import glob
from pathlib import Path
from fractions import Fraction
from scipy.signal import resample_poly
from denspp.offline import get_path_to_project
from denspp.offline.data_format.csv import CsvHandler
from denspp.offline.data_call.remote_handler import RemoteDownloader


@dataclass
class SettingsData:
    """Class for configuring the dataloader
    Attributes:
        pipeline:       String with name of the pipeline to use
        do_merge:       Boolean for run merge pipeline (True) or normal pipeline (False)
        path:           Path to data storage
        data_set:       String with key for used data set
        data_point:     Number within the dataset
        t_range_sec:    List of the given time range (in sec.) for cutting the data [x, y]
        ch_sel:         List of electrodes to use [empty=all]
        fs_resample:    Resampling frequency of the datapoint (== 0.0, no resampling)
        do_mapping:     Decision if mapping (if available) is used
        is_mapping_str: Boolean if mapping input from csv file is a string (True) or integer (false)
    """
    pipeline: str
    do_merge: bool
    path: str
    data_set: str
    data_case: int
    data_point: int
    t_range_sec: list[float]
    ch_sel: list[int]
    fs_resample: float
    do_mapping: bool
    is_mapping_str: bool


DefaultSettingsData = SettingsData(
    pipeline='PipelineV0',
    do_merge=False,
    path='data',
    data_set='',
    data_case=0, data_point=0,
    t_range_sec=[], ch_sel=[],
    fs_resample=50e3,
    do_mapping=True,
    is_mapping_str=False
)


class DataFromFile:
    """Class with data and meta information of the used neural dataset
    Attributes:
        data_name (str):        File name of used transient dataset
        data_type (str):        Type / Source of the transient dataset
        fs_orig (float):        Sampling rate of the original dataset [Hz]
        fs_used (float):        Sampling rate used in pipeline processing [Hz]
        time_end (float):       Float number with time duration of measurement / data [s]
        data_raw (np.ndarray):  Numpy array with data [shape=(num_elec (opt.), num_samples)]
        electrode_id (list):    List with integer values for matching data to electrode layout
        mapping_used (np.ndarray):  Array with 2D electrode mapping (loaded from external file)
        label_exist (bool):     Boolean if label with event position and ID exists
        evnt_xpos (list):       List with numpy arrays of event position on transient signal
        evnt_id (list):         List with numpy arrays of event ID for each electrode
    """
    # --- Meta Information
    data_name: str
    data_type: str
    fs_orig: float
    fs_used: float=0.0
    time_end: float
    # --- Raw data and Electrode Mapping
    data_raw: np.ndarray
    electrode_id: list
    mapping_used: np.ndarray
    # --- GroundTruth for Event Signal Processing
    label_exist: bool=False
    evnt_xpos: list
    evnt_id: list


class ControllerData:
    __logger: Logger
    __download_handler: RemoteDownloader
    _raw_data: DataFromFile
    _settings: SettingsData
    _path2folder_remote: str=''
    _path2folder_local: str=''
    _path2mapping: str=''

    def __init__(self) -> None:
        """Class for loading and manipulating the used dataset"""
        self._methods_available = dir(ControllerData)
        self.__logger = getLogger(__name__)
        self.__fill_factor = 1.
        self.__default_data_path = join(get_path_to_project(), 'data')
        self.__config_data_selection = [self.__default_data_path, 0, 0]
        self.__download_handler = RemoteDownloader()

    @staticmethod
    def _extract_func(class_obj: object) -> list:
        return  [method for method in dir(class_obj) if class_obj.__name__ in method]

    def do_cut(self) -> None:
        """Cutting all transient electrode signals in the given range"""
        if self._raw_data.fs_used == 0.0:
            self._raw_data.fs_used = self._raw_data.fs_orig

        # --- Getting the positition of used time range
        t_range = np.array(self._settings.t_range_sec)
        assert len(self._settings.t_range_sec) in [0, 2], f"t_range should be empty or have a length of 2 (not {len(self._settings.t_range_sec)})"
        if t_range.size == 2:
            rawdata_in = self._raw_data.data_raw

            idx0 = int(t_range[0] * self._raw_data.fs_used)
            idx1 = int(t_range[1] * self._raw_data.fs_used)

            # --- Cutting labeled information
            if self._raw_data.label_exist:
                evnt_pos_new = list()
                evnt_id_new = list()
                for evnt_pos, evnt_id in zip(self._raw_data.evnt_xpos, self._raw_data.evnt_id):
                    # Adapting new data
                    pos_start = np.argwhere(evnt_pos >= idx0).flatten()
                    idx2 = int(pos_start[0]) if pos_start.size > 0 else -1
                    pos_stopp = np.argwhere(evnt_pos <= idx1).flatten()
                    idx3 = int(pos_stopp[-1]) if pos_stopp.size > 0 else -1

                    if idx2 == -1 or idx3 == -1:
                        evnt_pos_new.append(list())
                        evnt_id_new.append(list())
                    else:
                        evnt_pos_new = evnt_pos[idx2:idx3] - idx0
                        evnt_id_new = evnt_id[idx2:idx3]
                self._raw_data.evnt_xpos = evnt_pos_new
                self._raw_data.evnt_id = evnt_id_new

            # --- Return adapted data
            self._raw_data.data_raw = rawdata_in[:, idx0:idx1] if len(rawdata_in.shape) == 2 else rawdata_in[idx0:idx1]
            self._raw_data.time_end = float(self._raw_data.data_raw.shape[-1] / self._raw_data.fs_used)
            self.__fill_factor = self._raw_data.data_raw.shape[-1] / rawdata_in.shape[-1]

    def do_resample(self, u_safe: float = 5e-6, num_points_mean: int=10) -> None:
        """Do resampling of all transient signals incl. label information
        :param u_safe:          Voltage range as safety window  for resampling
        :param num_points_mean: Integer number of points to average over
        :return:                None
        """
        desired_fs = self._settings.fs_resample
        do_resampling = bool(desired_fs != self._raw_data.fs_orig) and desired_fs != 0.0

        if do_resampling:
            self._raw_data.fs_used = desired_fs
            (p, q) = Fraction(self._raw_data.fs_used / self._raw_data.fs_orig).limit_denominator(10000).as_integer_ratio()
            scaling_value = p / q

            # --- Resampling the input
            data_out = list()
            for data_in in self._raw_data.data_raw:
                u_chck = np.mean(data_in[0:num_points_mean+1])
                du = u_chck if np.abs(u_chck) > u_safe else 0.0
                data_out.append(du + resample_poly(data_in - du, p, q))

            # --- Resampling the labeled information
            if self._raw_data.label_exist:
                spike_out = list()
                for evnt_pos in self._raw_data.evnt_xpos:
                    spike_out.append(np.array(scaling_value * evnt_pos, dtype=int))
            else:
                spike_out = list()

            self.__logger.debug(f"Resampling done from {self._raw_data.fs_orig} to {self._raw_data.fs_used} ({100 * scaling_value:.2f} %)")
            self._raw_data.data_raw = np.array(data_out, dtype=self._raw_data.data_raw.dtype)
            self._raw_data.evnt_xpos = spike_out
        else:
            self._raw_data.fs_used = self._raw_data.fs_orig

    def output_meta(self) -> None:
        """Print some meta information into the console"""
        self.__logger.info(f"... using data set of: {self._raw_data.data_name}")
        if not self._raw_data.fs_used == 0 and not self._raw_data.fs_used == self._raw_data.fs_orig:
            fs_addon = f" (resampling to {int(1e-3 * self._raw_data.fs_used)} kHz)"
        else:
            fs_addon = ""
        self.__logger.info(f"... original sampling rate of {int(1e-3 * self._raw_data.fs_orig)} kHz{fs_addon}")
        self.__logger.info(f"... using {self.__fill_factor * 100:.2f}% of the data (time length of {self._raw_data.time_end / self.__fill_factor:.2f} s)")

        if self._raw_data.label_exist:
            num_spikes = sum([val.size for val in self._raw_data.evnt_xpos])
            cluster_id = np.array([id for id in self._raw_data.evnt_id]).flatten()
            cluster_no = np.unique(cluster_id)
            self.__logger.info(f"... includes labels (noSpikes: {num_spikes} - noCluster: {cluster_no.size})")
        else:
            self.__logger.info(f"... has no labels / groundtruth")

    def get_data(self) -> DataFromFile:
        """Calling the raw data with optional ground truth
        :return:    Class DataHandler with raw data and meta information
        """
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
            folder2search = [folder for folder in glob(join(self.__config_data_selection[0], '*')) if folder_name in folder]
            if not len(folder2search):
                raise FileNotFoundError(f"Folder with index {folder2search} not found (locally and remotely)!")
            else:
                self._path2folder_local = folder2search[0]
                folder_content = glob(join(folder2search[0], data_type))
                if len(folder_content) == 0:
                    # --- Go into next folder structure and look there
                    folder_structure = glob(join(folder2search[0], '*'))
                    folder2search = join(folder2search[0], self.__config_data_selection[1]) if type(
                        self.__config_data_selection[1]) == str else join(
                        folder_structure[self.__config_data_selection[1]])
                    folder_content = glob(join(folder2search[0], data_type))
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
        else:
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

    def _prepare_access_file(self, folder_name: str, data_type: str) -> str:
        """Getting the file of the corresponding trial
        :param folder_name:     String with folder name where the data is located
        :param data_type:       String with data type of the data
        :return:                String with path to file
        """
        used_datapath = self.__default_data_path if self._settings.path == '' else self._settings.path
        self.__config_data_selection = [used_datapath, self._settings.data_case, self._settings.data_point]

        path2remote = self.__get_path_available_remote(folder_name, data_type)
        path2local = self.__get_data_available_local(path2remote, folder_name, data_type)
        if path2local:
            return path2local
        elif path2remote and not path2local:
            path2data = join(self._settings.path, dirname(path2remote[1:]))
            path2file = join(self._settings.path, path2remote[1:])
            makedirs(path2data, exist_ok=True)
            self.__download_handler.download_file(use_dataset=False, file_name=path2remote, destination_download=path2file)
            return path2file
        else:
            raise FileNotFoundError("--- File is not available. Please check! ---")

    def build_mapping(self, path2csv: str= '', index_search: str= 'Mapping_*.csv') -> None:
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
            self._raw_data.mapping_used = self._generate_electrode_mapping_from_csv()
        else:
            self.__logger.info("No electrode mapping is available")


    def _generate_electrode_mapping_from_csv(self) -> np.ndarray:
        """Function for reading the CSV file for electrode mapping of used (non-electrodes = 0)"""
        mapping_from_csv = CsvHandler(
            path=dirname(self._path2mapping),
            file_name=basename(self._path2mapping),
            delimiter=';'
        ).read_data_from_csv(
            include_chapter_line=False,
            start_line=0,
            type_load=str if self._settings.is_mapping_str else int
        ).tolist()

        # --- Generating numpy array
        mapping_dimension = [len(mapping_from_csv), len(mapping_from_csv[0])]
        electrode_mapping = np.zeros((mapping_dimension[0], mapping_dimension[1]), dtype=int)
        for row_idx, elec_row in enumerate(mapping_from_csv):
            for col_idx, elec_id in enumerate(elec_row):
                if row_idx < electrode_mapping.shape[0] and col_idx < electrode_mapping.shape[1]:
                    if type(elec_id) == str:
                        val = [idx + 1 for idx, val in enumerate(self._raw_data.electrode_id) if val == elec_id]
                        electrode_mapping[row_idx, col_idx] = val[0] if len(val) else 0
                    else:
                        electrode_mapping[row_idx, col_idx] = elec_id
        return electrode_mapping

    def _load_rawdata_into_pipeline(self, elec_type: str, dataset_name: str, file_name: str, fs_orig: float,
                                    elec_orn: list, rawdata: np.ndarray, scale_data: float,
                                    evnt_pos: list=(), evnt_id: list=()) -> None:
        """Function for preparing the loaded rawdata for using in pipeline process
        :param elec_type:   String with type description of the transient data
        :param file_name:   String with used file name
        :param fs_orig:     Sampling rate of transient signal [Hz]
        :param elec_orn:    List with Electrode orientation / mapping to rawdata
        :param rawdata:     Numpy array with raw data (shape: [num_channels, num_samples])
        :param scale_data:  Scaling factor for rawdata
        :param evnt_pos:    List with numpy arrays of the event positions (should have same length like elec_orn)
        :param evnt_id:     List with numpy arrays of the event ID (should have same length like elec_orn)
        :return:            None
        """
        if len(rawdata.shape) == 2:
            assert rawdata.shape[0] == len(elec_orn), "Variable rawdata must have two dimensions (num_channels, num_samples)"
        else:
            assert len(elec_orn) == 1, "Variable rawdata has one dimension with (num_samples, ), also elec_orn should have a length of 1"
        if len(evnt_pos):
            assert type(evnt_pos) == list, "Variable evnt_pos must have type list"
            assert len(evnt_pos) == len(elec_orn), "Length of event_pos should have same length like num_electrodes"
            assert type(evnt_id) == list, "Variable evnt_id must have type list"
            assert len(evnt_id) == len(elec_orn), "Length of event_pos should have same length like num_electrodes"

        self._raw_data = DataFromFile()
        # --- Including meta data
        self._raw_data.data_type = elec_type
        self._raw_data.data_name = dataset_name + Path(file_name).stem
        self._raw_data.fs_orig = fs_orig
        # --- Including raw data
        self._raw_data.electrode_id = elec_orn
        # --- Scaling the data in place
        rawdata = rawdata.astype(np.float32)
        rawdata *= scale_data
        self._raw_data.data_raw = rawdata
        if len(rawdata.shape) == 2:
            pass
        else:
            self._raw_data.data_raw = np.expand_dims(self._raw_data.data_raw, axis=0)

        self._raw_data.time_end = self._raw_data.data_raw.shape[1] / self._raw_data.fs_orig
        # --- Including labels
        if len(evnt_pos) and len(evnt_id):
            self._raw_data.evnt_xpos = evnt_pos
            self._raw_data.evnt_id = evnt_id
            self._raw_data.label_exist = True
        else:
            self._raw_data.evnt_xpos = list()
            self._raw_data.evnt_id = list()
            self._raw_data.label_exist = False

    def do_call(self, *args) -> None:
        """Loading the dataset
        :return:    None
        """
        method_to_call = None
        for method in self._methods_available:
            if self._settings.data_set in method:
                method_to_call = method
                break

        if not self._settings.data_set or not method_to_call:
            # --- Getting the function to call
            warning_text = "\nPlease select key words in variable 'data_set' for calling methods to read transient data"
            warning_text += "\n=========================================================================================="
            for method in self._methods_available:
                warning_text += f"\n\t{method}"

            self.__logger.error(warning_text)
            raise ValueError
        else:
            getattr(self, method_to_call)(*args)
