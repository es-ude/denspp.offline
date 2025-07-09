import numpy as np
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
        """Loading 1d-test data without getting files"""
        fs_used = 20e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_1d",
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1],
            rawdata=np.random.randn(int(fs_used)),
            scale_data=1e-6,
        )

    def __load_test_2d(self) -> None:
        """Loading 2d-test data without getting files"""
        fs_used = 20e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_2d",
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1, 2, 3, 4],
            rawdata=np.random.randn(4, int(fs_used)),
            scale_data=1e-6,
        )

    def __load_test_2d_zero(self) -> None:
        """Loading 2d-test data without getting files"""
        fs_used = 20e3
        self._load_rawdata_into_pipeline(
            elec_type="Test_2d",
            file_name='',
            fs_orig=fs_used,
            elec_orn=[1, 2, 3],
            rawdata=np.random.randn(3, int(fs_used)),
            scale_data=1e-6,
        )

    def __load_test_args(self, fs: float, data: np.ndarray) -> None:
        self._load_rawdata_into_pipeline(
            elec_type="Test_args",
            file_name='',
            fs_orig=fs,
            elec_orn=[1],
            rawdata=data,
            scale_data=1.,
        )