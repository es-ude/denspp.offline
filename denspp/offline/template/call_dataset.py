import numpy as np
from logging import getLogger, Logger
from denspp.offline.data_format import JsonHandler
from denspp.offline.data_call import build_waveform_dataset, SettingsWaveformDataset, DefaultSettingsWaveformDataset
from denspp.offline.dnn import DatasetFromFile
from denspp.offline.dnn.data_config import SettingsDataset, ControllerDataset
from denspp.offline.dnn.data_processor import DataProcessor


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
        dataset = DatasetFromFile(
            data=data.reshape(-1, 28, 28),
            label=np.array(label, dtype=np.uint8),
            dict=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
            mean=np.zeros(shape=(10, 28, 28)),
        )
        return self._processor.process_vision_datasets(
            data=dataset
        )

    def __get_sinusoidal(self) -> None:
        pass

    def __prepare_sinusoidal(self) -> DatasetFromFile:
        seq_len = 100
        n_samples = 2000
        noise_amp = 0.5

        data = []
        labels = []
        window = np.linspace(start=0, stop=2 * np.pi, num=seq_len)
        for _ in range(n_samples):
            if np.random.rand() > 0.5:
                x = np.sin(window)
                label = 0
            else:
                x = np.cos(window)
                label = 1
            x += noise_amp * np.random.randn(seq_len)  # kleines Rauschen
            data.append(x)  # shape: (seq_len, 1)
            labels.append(label)

        dataset = DatasetFromFile(
            data=np.array(data, dtype=float),
            label=np.array(labels, dtype=int),
            dict=['sin', 'cos'],
            mean=np.zeros_like(np.array(data))
        )
        return self._processor.process_timeseries_datasets(data=dataset, add_noise_cluster=False)

    def __get_waveforms(self) -> SettingsWaveformDataset:
        return JsonHandler(
            template=DefaultSettingsWaveformDataset,
            path='config',
            file_name='Config_WaveformDataset'
        ).get_class(SettingsWaveformDataset)

    def __prepare_waveforms(self) -> DatasetFromFile:
        data = build_waveform_dataset(
            settings_data=self.__get_waveforms()
        )
        dataset = DatasetFromFile(
            data=data.data,
            label=data.label,
            dict=data.dict,
            mean=np.zeros(shape=(len(data.dict), *data.data.shape[1:]))
        )
        return self._processor.process_timeseries_datasets(data=dataset)
