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
        folder_name = "E0_Kirchner2023"
        meta_type = 'Metadata_*.txt'
        makrer_type = 'Markerfile_*.txt'
        data_type = '*set*.txt'
        self._prepare_access_file(folder_name, data_type, 0)

        # Read textfile and convert
        file = open(self.path2file, 'r')
        loaded_data = list()
        for line in file:
            input = line.split(" ")
            data0 = list()
            for val in input:
                if val:
                    data0.append(float(val))
            loaded_data.append(data0)

        loaded_data = np.array(loaded_data[:-1])
        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Synthetic"
        data.noChannel = loaded_data.shape[1]
        data.gain = 1
        data.fs_orig = int(1 / 1000)
        data.raw_data = [(data.gain * loaded_data)]
        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = False
        # Return
        self.raw_data = data
