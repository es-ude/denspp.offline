import numpy as np
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
        self.__load_Kirchner2023(0, 0)

    def __load_Kirchner2023(self, case: int, point: int) -> None:
        """Loading EMG recording files from Kirchner (2023)"""
        folder_name = "01_Kirchner2023"
        meta_type = 'Metadata_*.txt'
        makrer_type = 'Markerfile_*.txt'
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
        data.data_type = "Synthetic"
        data.noChannel = data_used.shape[0]
        data.gain = 1
        data.fs_orig = int(1 / 1000)
        data.raw_data = [data.gain * data_used[idx, :] for idx in range(num_channels)]
        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = False
        # Return
        self.raw_data = data
