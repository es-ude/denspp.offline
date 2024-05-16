import glob

import numpy as np
from os.path import join
from package.data_call.call_handler import DataController, SettingsDATA


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
class DataLoader(DataController):
    """Class for loading and manipulating the used dataset"""
    raw_data: DataHandler
    settings: SettingsDATA
    path2file: str

    def __init__(self, setting: SettingsDATA) -> None:
        DataController.__init__(self)
        self.settings = setting
        self.select_electrodes = list()
        self.path2file = str()

    def do_call(self):
        """"""
        self._prepare_call()
        # --- Data Source Selection
        self.__load_Kirchner2023(0, 0)

        # --- Post-Processing
        self._transform_rawdata_to_numpy()
        self._transform_rawdata_mapping(True, [])

    def __load_Kirchner2023(self, case: int, point: int) -> None:
        """Loading EMG recording files from Kirchner (2023)"""
        folder_name = "01_Kirchner2023"
        marker_type = 'Markerfile_*.txt'
        data_type = '*set*.txt'
        self._prepare_access_file(folder_name, data_type, 0)

        # --- Read textfile and convert
        num_channels = 10
        loaded_data = [[] for idx in range(num_channels)]

        file = open(self.path2file, 'r')
        for line in file:
            input = line.split(" ")
            sel_list = 0
            for val in input:
                if val:
                    loaded_data[sel_list].append(float(val))
                    sel_list += 1
        del file, input, sel_list

        # --- Read markerfile
        path2data = join(self.settings.path, folder_name, marker_type)
        list_marker = glob.glob(path2data)
        loaded_marker = list()
        loaded_type = list()
        file = open(list_marker[0], 'r')
        for idx, line in enumerate(file):
            if idx > 6:
                input = line.split(",")
                loaded_marker.append(int(input[1][:-1]))
                match input[0]:
                    case "S 64":
                        id = 0
                    case "S 32":
                        id = 1
                    case "S 48":
                        id = 2
                    case "S 96":
                        id = 3
                    case "S 80":
                        id = 4
                    case _:
                        id = -1
                loaded_type.append(id)
        del input

        # --- Bringing data into right format
        num_channels = len(loaded_data)
        num_samples = np.zeros((num_channels,), dtype=int)
        for idx, data0 in enumerate(loaded_data):
            num_samples[idx] = len(data0)

        data_used = np.zeros((num_channels, num_samples.min()), dtype=float)
        for idx, data0 in enumerate(loaded_data):
            data_used[idx, :] = np.array(data0[0:num_samples.min()])
        del loaded_data, data0

        # --- Processing data
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
        data.evnt_dict = ["S64", "S32", "S48", "S96", "S80"]
        # Return
        self.raw_data = data
