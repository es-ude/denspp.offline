import numpy as np
from logging import getLogger, Logger
from denspp.offline.data_format import YamlHandler
from denspp.offline.data_call import build_waveform_dataset, SettingsWaveformDataset, DefaultSettingsWaveformDataset
from denspp.offline.dnn import DatasetFromFile
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, ControllerDataset
from denspp.offline.dnn.processor import DataProcessor


class DatasetLoader(ControllerDataset):
    _logger: Logger
    _settings: SettingsDataset
    _processor: DataProcessor
    _path: str

    def __init__(self, settings: SettingsDataset, temp_folder: str='') -> None:
        """Class for downloading (function name with '__get_xyz')
        and preparing (function name with '__prepare_xyz') custom-defined datasets to train deep learning models
        :param settings:  Object of class SettingsDataset for handling dataset used in DeepLearning"""
        super().__init__(settings, temp_folder)
        self._logger = getLogger(__name__)
        self._processor = DataProcessor(settings)

    def __get_mnist(self) -> None:
        pass

    def __prepare_mnist(self) -> DatasetFromFile:
        from sklearn.datasets import fetch_openml
        data, label = fetch_openml("mnist_784", return_X_y=True, as_frame=False, parser="liac-arff")
        data_process = self._processor.process_vision_datasets(
            picture=data.reshape(-1, 28, 28),
            label=label
        )
        return DatasetFromFile(
            data=data_process.data,
            label=data_process.label,
            dict=['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'],
            mean=data_process.mean,
        )

    def __get_waveforms(self) -> SettingsWaveformDataset:
        return YamlHandler(
            template=DefaultSettingsWaveformDataset,
            path='config',
            file_name='Config_WaveformDataset'
        ).get_class(SettingsWaveformDataset)

    def __prepare_waveforms(self) -> DatasetFromFile:
        data = build_waveform_dataset(
            settings_data=self.__get_waveforms()
        )
        return DatasetFromFile(
            data=data.data,
            label=data.label,
            dict=data.dict,
            mean=np.zeros(shape=(data.label.size, data.data.shape[1]))
        )
