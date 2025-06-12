import numpy as np
from denspp.offline.data_call.call_handler import ControllerData, SettingsData
from scipy.io import loadmat
from pyxdf import load_xdf


class DataLoader(ControllerData):
    _settings: SettingsData
    _methods_available: list

    def __init__(self, settings: SettingsData) -> None:
        """Class for loading and manipulating the used dataset
        :param settings: Settings class instance
        """
        ControllerData.__init__(self)
        self._settings = settings
        self._methods_available = self._extract_func(self.__class__)

    def __load_test_1d(self) -> None:
        """Loading 1d-test data without getting files"""
        fs_used = 10e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_1d",
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1],
            rawdata=np.random.randn(10000),
            scale_data=1e-6,
        )

    def __load_test_2d(self) -> None:
        """Loading 2d-test data without getting files"""
        fs_used = 10e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_2d",
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1, 2, 3, 4],
            rawdata=np.random.randn(4, 10000),
            scale_data=1e-6,
        )

    def __load_test_2d_zero(self) -> None:
        """Loading 2d-test data without getting files"""
        fs_used = 10e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_2d",
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1, 2, 3],
            rawdata=np.random.randn(3, 10000),
            scale_data=1e-6,
        )

    def __load_martinez_simulation(self) -> None:
        """Loading synthethic files from Quiroga simulation (2009)"""
        folder_name = "_SimDaten_Martinez2009"
        data_type = 'simulation_*.mat'
        path2file = self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(path2file)

        fs_used = float(1 / loaded_data["samplingInterval"][0][0] * 1000)
        spike_xoffset = int(-0.1e-3 * fs_used)
        self._load_rawdata_into_pipeline(
            elec_type="Synthetic",
            file_name=path2file,
            fs_orig=fs_used,
            elec_orn=[int(loaded_data["chan"][0]) - 1],
            rawdata=loaded_data["data"][0],
            scale_data=0.5e-6,
            evnt_pos=[loaded_data["spike_times"][0][0][0] - spike_xoffset],
            evnt_id=[loaded_data["spike_class"][0][0][0]]
        )

    def __load_pedreira_simulation(self) -> None:
        """Loading synthethic files from Quiroga simulator (2012)"""
        folder_name = "_SimDaten_Pedreira2012"
        data_type = 'simulation_*.mat'

        path2file = self._prepare_access_file(folder_name, data_type)
        path2label = self._prepare_access_file(folder_name, "ground_truth.mat")
        loaded_data = loadmat(path2file)
        ground_truth = loadmat(path2label)

        fs_used = 24e3
        spike_xoffset = int(-0.1e-6 * fs_used)
        num_index = int(path2file.split("_")[-1].split(".")[0])
        self._load_rawdata_into_pipeline(
            elec_type="Synthetic",
            file_name=path2file,
            fs_orig=fs_used,
            elec_orn=[int(loaded_data["data"].shape[0]) - 1],
            rawdata=loaded_data["data"][0],
            scale_data=25e-6,
            evnt_pos=[ground_truth["spike_first_sample"][0][num_index - 1][0] - spike_xoffset],
            evnt_id=[ground_truth["spike_classes"][0][num_index - 1][0]]
        )

    def __load_quiroga_simulation(self) -> None:
        """Loading synthetic recordings from Quiroga simulator (Common benchmark)"""
        folder_name = "_SimDaten_Quiroga2020"
        data_type = 'C_*.mat'
        path2file = self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(path2file, mat_dtype=True)

        fs_used = float(1000 / loaded_data["samplingInterval"][0][0])
        self._load_rawdata_into_pipeline(
            elec_type="Synthetic",
            file_name=path2file,
            fs_orig=fs_used,
            elec_orn=[int(loaded_data["chan"][0][0]) - 1],
            scale_data=100e-6,
            rawdata=loaded_data["data"][0],
            evnt_pos=[loaded_data["spike_times"][0][0][0] - int(-0.5e-6 * fs_used)],
            evnt_id=[loaded_data["spike_class"][0][0][0] - 1]
        )

    def __load_denspp_online(self) -> None:
        """Function for loading the *.xdf files from custom hardware readout with DeNSPP.online framework"""
        folder_name = "_Custom_Hardware"
        data_type = '*.xdf'
        path2file = self._prepare_access_file(folder_name, data_type)
        loaded_data = load_xdf(path2file)[0][0]

        fs_used = float(loaded_data['info']['nominal_srate'][0])
        self._load_rawdata_into_pipeline(
            elec_type=loaded_data['info']['name'],
            file_name=path2file,
            fs_orig=fs_used,
            elec_orn=np.arange(0, loaded_data['time_series'].shape[1]).tolist(),
            scale_data=1.0,
            rawdata=np.transpose(np.float32(loaded_data["data"][0], (1, 0)))
        )
