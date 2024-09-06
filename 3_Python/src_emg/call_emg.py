import glob
import numpy as np
from os.path import join

from package.data_call.call_handler import _DataController, SettingsDATA


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
        self.electrode_id = list()
        self.data_raw = list()
        self.data_mapping_avai = False
        # --- GroundTruth (per Channel)
        self.label_exist = False
        self.evnt_xpos = list()
        self.evnt_cluster_id = list()
        self.evnt_dict = list()


# ----- Read Settings -----
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

    def _transform_label_from_csv_to_numpy(self, marker_loaded: list, label_text: list) -> [list, list]:
        """"""
        loaded_type = list()
        for idx, label in enumerate(marker_loaded[0]):
            if idx > 6:
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

    def do_call(self):
        """"""
        self._prepare_call()
        # --- Data Source Selection
        match self.settings.data_set:
            case 0:
                self.__load_Kirchner2023()
            case 1:
                self.__load_Kirchner2024()

        # --- Post-Processing
        self._transform_rawdata_to_numpy()
        self._transform_rawdata_mapping(True, [])

    def __load_Kirchner2023(self) -> None:
        """Loading EMG recording files from E. Kirchner (2023) working with an orthese"""
        # --- Part #1: Loading data and label
        folder_name = "*_Kirchner_Orthese2023"
        marker_type = 'Markerfile_*.txt'
        data_type = '*set*.txt'
        self._prepare_access_file(folder_name, data_type)

        # Read textfile and converting
        num_channels = 10
        loaded_data_preload = self._read_csv_file(self.path2file, num_channels, " ")
        data_used = self._transform_rawdata_from_csv_to_numpy(loaded_data_preload)

        # Read markerfile and converting
        loaded_type_dict = ["S 64", "S 32", "S 48", "S 96", "S 80"]
        path2data = join(self.settings.path, folder_name, marker_type)
        list_marker = glob.glob(path2data)
        loaded_marker_preloaded = self._read_csv_file(list_marker[0], 2, ",")
        loaded_type, loaded_marker = self._transform_label_from_csv_to_numpy(loaded_marker_preloaded, loaded_type_dict)

        # --- Part #2: Applying data filter
        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Orthese"
        data.noChannel = data_used.shape[0]
        data.gain = 1
        data.data_fs_orig = int(1000)
        data.data_raw = [data.gain * data_used[idx, :] for idx in range(num_channels-1)]
        data.data_time = data.data_raw[0].shape[0] / data.data_fs_orig
        data.data_mapping_avai = False
        # Groundtruth
        data.label_exist = True
        data.evnt_xpos = [np.array(loaded_marker)-data_used[-1, 0] for idx in range(num_channels-1)]
        data.evnt_cluster_id = loaded_type
        data.evnt_dict = loaded_type_dict
        # Return
        self.raw_data = data

    def __load_Kirchner2024(self) -> None:
        """Loading EMG recording files from E. Kirchner (2024) working with myo system"""
        # --- Part #1: Loading data and label
        folder_name = "_Kirchner_Myo2024"
        data_type = '*set*.csv'
        self._prepare_access_file(folder_name, data_type)

        num_channels = 8
        data_preloaded = self._read_csv_file(self.path2file,  8, ",")
        data_used = self._transform_rawdata_from_csv_to_numpy(data_preloaded)

        # --- Part #2: Applying data filter
        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Myo_Armband"
        data.noChannel = data_used.shape[0]
        data.gain = 1
        data.data_fs_orig = int(1000)
        data.data_raw = [data.gain * data_used[idx, :] for idx in range(num_channels)]
        data.data_time = data.data_raw[0].shape[0] / data.data_fs_orig
        data.data_mapping_avai = False
        # Groundtruth
        data.label_exist = False
        # Return
        self.raw_data = data
