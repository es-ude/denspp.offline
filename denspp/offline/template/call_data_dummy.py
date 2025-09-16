import numpy as np
from scipy.io import loadmat
from denspp.offline.data_call import ControllerData, SettingsData


class DataLoaderTest(ControllerData):
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
        fs_used = 20e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_1d",
            dataset_name='',
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1],
            rawdata=np.random.randn(int(fs_used)),
            scale_data=1e-6,
        )

    def __load_test_2d(self) -> None:
        fs_used = 20e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_2d",
            dataset_name='',
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1, 2, 3, 4],
            rawdata=np.random.randn(4, int(fs_used)),
            scale_data=1e-6,
        )

    def __load_test_2d_zero(self) -> None:
        fs_used = 20e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_2d",
            dataset_name='',
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1, 2, 3],
            rawdata=np.random.randn(3, int(fs_used)),
            scale_data=1e-6,
        )

    def __load_test_args(self, fs: float, data: np.ndarray) -> None:
        self._load_rawdata_into_pipeline(
            elec_type="Test_args",
            dataset_name='',
            file_name='',
            fs_orig=fs,
            elec_orn=[1],
            rawdata=data,
            scale_data=1.,
        )

    def __load_martinez_with_labels(self) -> None:
        path2file = self._prepare_access_file(folder_name="_SimDaten_Martinez2009", data_type='simulation_*.mat')
        loaded_data = loadmat(path2file)

        fs_used = float(1 / loaded_data["samplingInterval"][0][0] * 1000)
        spike_xoffset = int(-0.1e-3 * fs_used)
        self._load_rawdata_into_pipeline(
            elec_type="Synthetic",
            dataset_name='martinez',
            file_name=path2file,
            fs_orig=fs_used,
            elec_orn=[int(loaded_data["chan"][0]) - 1],
            rawdata=loaded_data["data"][0],
            scale_data=0.5e-6,
            evnt_pos=[loaded_data["spike_times"][0][0][0] - spike_xoffset],
            evnt_id=[loaded_data["spike_class"][0][0][0]]
        )

    def __load_martinez_without_labels(self) -> None:
        path2file = self._prepare_access_file(folder_name="_SimDaten_Martinez2009", data_type='simulation_*.mat')
        loaded_data = loadmat(path2file)

        fs_used = float(1 / loaded_data["samplingInterval"][0][0] * 1000)
        self._load_rawdata_into_pipeline(
            elec_type="Synthetic",
            dataset_name='martinez',
            file_name=path2file,
            fs_orig=fs_used,
            elec_orn=[int(loaded_data["chan"][0]) - 1],
            rawdata=loaded_data["data"][0],
            scale_data=0.5e-6
        )
