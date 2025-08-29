import numpy as np
from os import makedirs
from os.path import join, exists
from shutil import copy
from datetime import datetime
from logging import getLogger, Logger
from denspp.offline import get_path_to_project
from denspp.offline.dnn.model_library import ModuleRegistryManager


class PipelineLibrary:
    """Class for searching all Pipeline Processors in repository to get an overview"""
    def get_registry(self, package: str="src_pipe") -> ModuleRegistryManager:
        m = ModuleRegistryManager(r"\bPipelineV\d+(?:Merge)?\b")
        chck = exists(join(get_path_to_project(), package))
        m.register_package(package) if chck else m.register_package("denspp.offline.template")
        return m


class DataloaderLibrary:
    """Class for searching all Pipeline Processors in repository to get an overview"""
    def get_registry(self, package: str="src_pipe") -> ModuleRegistryManager:
        m = ModuleRegistryManager(r"\bDataLoader(Test)?\b")
        chck = exists(join(get_path_to_project(), package))
        m.register_package(package) if chck else m.register_package("denspp.offline.template")
        return m


class PipelineCMD:
    """Class for handling the pipeline processing"""
    path2save: str=''
    _path2pipe: str=''
    _path2start: str=get_path_to_project()
    _logger: Logger=getLogger(__name__)

    def get_pipeline_name(self) -> str:
        """Getting the name of the pipeline"""
        return self.__class__.__name__

    def generate_run_folder(self, path2runs: str, addon: str) -> None:
        """Generating the default folder for saving figures and data
        :param path2runs:   Main folder in which the figures and data is stored
        :param addon:       Name of new folder for saving results
        :return:            None
        """
        str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'{str_datum}_{self.get_pipeline_name().lower()}{addon}'
        path2start = join(self._path2start, path2runs)
        path2save = join(path2start, folder_name)
        makedirs(path2save, exist_ok=True)
        if self._path2pipe:
            copy(src=self._path2pipe, dst=path2save)
        self.path2save = path2save
        self._logger.debug(f"Creating run folder and copying the pipeline into: {path2save}")

    def apply_mapping(self, data: np.ndarray, electrode_id: list, mapping: np.ndarray) -> np.ndarray:
        """Transforming the input data to 2D array using electrode mapping configuration
        :param data:            Input data with shape (num_channels, num_samples)
        :param electrode_id:    List with name/numbers of electrodes used on data
        :param mapping:         Numpy array with electrode ID localisation
        :return:                Numpy array with transformed data to 2D
        """
        assert len(data.shape) == 2 and data.shape[0] > 1, "Shape of input data must higher than 2"
        assert data.shape[0] == len(electrode_id), "Mismatch between electrode_id and data shape"
        assert np.count_nonzero(mapping) == len(electrode_id), "No mapping is available or mapping content does not match electrode ID"

        dut = np.zeros((mapping.shape[0], mapping.shape[1], data.shape[-1]), dtype=data.dtype)
        for row_idx in range(0, mapping.shape[0]):
            for col_idx in range(0, mapping.shape[1]):
                if mapping[row_idx, col_idx] > 0:
                    use_data_id = 0
                    for channel in electrode_id:
                        if channel == mapping[row_idx, col_idx]:
                            dut[row_idx, col_idx, :] = data[use_data_id, :]
                            break
                        use_data_id += 1
        self._logger.info("... transforming raw data array from 1D to 2D")
        return dut

    def deploy_mapping(self, data: np.ndarray, electrode_id: list, mapping: np.ndarray) -> np.ndarray:
        """Transforming the 2D data to normal electrode orientation using electrode mapping configuration
        :param data:            Input data with shape (num_rows, num_cols, num_samples)
        :param electrode_id:    List with name/numbers of electrodes used on data
        :param mapping:         Numpy array with electrode ID localisation
        :return:                Numpy array with original data format
        """
        assert len(data.shape) == 3, "Shape of input data must higher than 2"
        assert data.shape[0] * data.shape[1] >= len(electrode_id), "Mismatch between electrode_id and data shape"
        assert len(mapping), "No mapping is available"

        dut = np.zeros((len(electrode_id), data.shape[-1]), dtype=data.dtype)
        for row_idx in range(0, mapping.shape[0]):
            for col_idx in range(0, mapping.shape[1]):
                if mapping[row_idx, col_idx] > 0:
                    for channel in electrode_id:
                        if channel == mapping[row_idx, col_idx]:
                            dut[channel-1, :] = data[row_idx, col_idx, :]
                            break
        self._logger.info("... transforming raw data array from 2D to 1D")
        return dut

    def save_results(self, name: str, data: dict) -> None:
        """Saving the data with a dictionary
        :param name:    File name for saving results
        :param data:    Dictionary with data content
        :return:        None
        """
        path2data = join(self.path2save, name)
        np.save(path2data, data, allow_pickle=True)
        self._logger.info(f"... data saved in: {path2data}")
