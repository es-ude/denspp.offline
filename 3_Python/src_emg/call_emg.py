import glob
import numpy as np
from os.path import join
from package.data_call.call_handler import _DataController, SettingsDATA, DataHandler, transform_label_from_csv_to_numpy


class DataLoader(_DataController):
    """Class for loading and manipulating the used dataset"""
    raw_data: DataHandler
    settings: SettingsDATA
    path2file: str

    def __init__(self, setting: SettingsDATA) -> None:
        _DataController.__init__(self)
        self.settings = setting
        self.select_electrodes = list()
        self.path2file = str()
        self._methods_available = dir(DataLoader)

    def __load_method00_kirchner2023(self) -> None:
        """Loading EMG recording files from E. Kirchner (2023) working with an orthese"""
        # --- Part #1: Loading data and label
        folder_name = "*_Kirchner_Orthese2023"
        marker_type = 'Markerfile_*.txt'
        data_type = '*set*.txt'
        self._prepare_access_file(folder_name, data_type)
        loaded_data_preload = self._read_csv_file(self.path2file, 10, " ")
        data_used = self._transform_rawdata_from_csv_to_numpy(loaded_data_preload)

        # Read markerfile and converting
        loaded_type_dict = ["S 64", "S 32", "S 48", "S 96", "S 80"]
        path2data = join(self.settings.path, folder_name, marker_type)
        list_marker = glob.glob(path2data)
        loaded_marker_preloaded = self._read_csv_file(list_marker[0], 2, ",")
        loaded_type, loaded_marker = transform_label_from_csv_to_numpy(loaded_marker_preloaded, loaded_type_dict, 6)

        # --- Part #2: Applying data filter
        self.raw_data = DataHandler()
        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Orthese"
        self.raw_data.data_fs_orig = int(1000)
        # Electrode Mapping Design
        self.raw_data.data_mapping_avai = False
        self.raw_data.mapping_dimension = [1, 10]
        # Raw data
        self.raw_data.data_raw = [1.0 * data_used[idx, :] for idx in range(data_used.shape[0]-1)]
        self.raw_data.data_time = self.raw_data.data_raw[0].shape[0] / self.raw_data.data_fs_orig
        # Groundtruth
        self.raw_data.label_exist = True
        self.raw_data.evnt_xpos = [np.array(loaded_marker)-data_used[-1, 0] for idx in range(data_used.shape[0]-1)]
        self.raw_data.evnt_cluster_id = loaded_type
        self.raw_data.evnt_dict = loaded_type_dict

    def __load_method01_kirchner2024(self) -> None:
        """Loading EMG recording files from E. Kirchner (2024) working with myo system"""
        # --- Part #1: Loading data and label
        folder_name = "_Kirchner_Myo2024"
        data_type = '*set*.csv'
        self._prepare_access_file(folder_name, data_type)
        data_preloaded = self._read_csv_file(self.path2file,  8, ",")
        data_used = self._transform_rawdata_from_csv_to_numpy(data_preloaded)

        # --- Part #2: Applying data filter
        self.raw_data = DataHandler()
        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Myo_Armband"
        self.raw_data.data_fs_orig = int(1000)
        # Electrode Design Information
        self.raw_data.data_mapping_avai = False
        self.raw_data.mapping_dimension = [1, 8]
        # Raw data
        self.raw_data.data_raw = [1.0 * data_used[idx, :] for idx in range(data_used.shape[0])]
        self.raw_data.data_time = self.raw_data.data_raw[0].shape[0] / self.raw_data.data_fs_orig
        # Groundtruth
        self.raw_data.label_exist = False
