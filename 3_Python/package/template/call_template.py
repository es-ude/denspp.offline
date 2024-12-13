import numpy as np
from os.path import join
from scipy.io import loadmat
from package.data_call.call_handler import _DataController, SettingsDATA, DataHandler


class DataLoader(_DataController):
    """Class for loading and manipulating the used dataset"""
    _raw_data: DataHandler
    _settings: SettingsDATA
    _path2file: str

    def __init__(self, setting: SettingsDATA) -> None:
        _DataController.__init__(self)
        self._settings = setting
        self.select_electrodes = list()
        self._path2file = str()
        self._methods_available = dir(DataLoader)

    def __load_martinez_simulation(self) -> None:
        """Loading synthethic files from Quiroga simulation (2009)"""
        folder_name = "_SimDaten_Martinez2009"
        data_type = 'simulation_*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self._path2file)

        self._raw_data = DataHandler()
        # Meta information
        self._raw_data.data_name = folder_name
        self._raw_data.data_type = "Synthetic"
        self._raw_data.data_fs_orig = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
        self._raw_data.device_id = [0]
        # Electrode mapping information
        self._raw_data.mapping_exist = False
        self._raw_data.mapping_dimension = [1, 1]
        # Input
        self._raw_data.electrode_id = [int(loaded_data["chan"][0]) - 1]
        self._raw_data.data_raw = [0.5e-6 * np.float32(loaded_data["data"][0])]
        self._raw_data.data_time = loaded_data["data"][0].size / self._raw_data.data_fs_orig
        # Groundtruth
        self._raw_data.label_exist = True
        spike_xoffset = int(-0.1e-3 * self._raw_data.data_fs_orig)
        self._raw_data.evnt_xpos = [(loaded_data["spike_times"][0][0][0] - spike_xoffset)]
        self._raw_data.evnt_id = [(loaded_data["spike_class"][0][0][0])]
        # Behaviour
        self._raw_data.behaviour_exist = False
        self._raw_data.behaviour = None
        del loaded_data

    def __load_pedreira_simulation(self) -> None:
        """Loading synthethic files from Quiroga simulator (2012)"""
        folder_name = "_SimDaten_Pedreira2012"
        data_type = 'simulation_*.mat'
        self._prepare_access_file(folder_name, data_type)

        prep_index = self._path2file.split("_")[-1]
        num_index = int(prep_index[0:2])
        path2label = join(self._settings.path, folder_name, "ground_truth.mat")

        loaded_data = loadmat(self._path2file)
        ground_truth = loadmat(path2label)

        self._raw_data = DataHandler()
        # Meta information
        self._raw_data.data_name = folder_name
        self._raw_data.data_type = "Synthetic"
        self._raw_data.data_fs_orig = 24e3
        self._raw_data.device_id = [0]
        # Electrode mapping information
        self._raw_data.mapping_exist = False
        self._raw_data.mapping_dimension = [1, 1]
        # Raw data
        self._raw_data.electrode_id = [int(loaded_data["data"].shape[0]) - 1]
        self._raw_data.data_raw = [25e-6 * np.float32(loaded_data["data"][0])]
        self._raw_data.data_time = loaded_data["data"].shape[1] / self._raw_data.data_fs_orig
        # Groundtruth
        self._raw_data.label_exist = True
        spike_xoffset = int(-0.1e-6 * self._raw_data.data_fs_orig)
        self._raw_data.evnt_xpos = [(ground_truth["spike_first_sample"][0][num_index - 1][0] - spike_xoffset)]
        self._raw_data.evnt_id = [(ground_truth["spike_classes"][0][num_index - 1][0])]
        # Behaviour
        self._raw_data.behaviour_exist = False
        self._raw_data.behaviour = None
        del loaded_data

    def __load_quiroga_simulation(self) -> None:
        """Loading synthetic recordings from Quiroga simulator (Common benchmark)"""
        folder_name = "_SimDaten_Quiroga2020"
        data_type = 'C_*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self._path2file, mat_dtype=True)

        self._raw_data = DataHandler()
        # --- Input and meta
        self._raw_data.data_name = folder_name
        self._raw_data.data_type = "Synthetic"
        self._raw_data.data_fs_orig = float(1000 / loaded_data["samplingInterval"][0][0])
        self._raw_data.device_id = [0]
        # Electrode mapping information
        self._raw_data.mapping_exist = False
        self._raw_data.mapping_dimension = [1, 1]
        # Input data
        self._raw_data.electrode_id = [int(loaded_data["chan"][0][0]) - 1]
        self._raw_data.data_raw = [100e-6 * np.float32(loaded_data["data"][0])]
        self._raw_data.data_time = loaded_data["data"].shape[1] / self._raw_data.data_fs_orig
        # --- Groundtruth
        self._raw_data.label_exist = True
        spike_xoffset = int(-0.5e-6 * self._raw_data.data_fs_orig)
        self._raw_data.evnt_xpos = [(loaded_data["spike_times"][0][0][0] - spike_xoffset)]
        self._raw_data.evnt_id = [(loaded_data["spike_class"][0][0][0] - 1)]
        # Behaviour
        self._raw_data.behaviour_exist = False
        self._raw_data.behaviour = None
        del loaded_data
