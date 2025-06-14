import numpy as np
from logging import getLogger, Logger
from dataclasses import dataclass
from os import makedirs
from os.path import join, abspath, isabs, exists
from torch import Tensor, concat, from_numpy
from torchvision import datasets, transforms
from denspp.offline.yaml_handler import YamlHandler
from denspp.offline import get_path_to_project_start
from denspp.offline.data_call.call_cellbib import CellSelector
from denspp.offline.data_call.owncloud_handler import OwnCloudDownloader
from denspp.offline.data_generator import SettingsWaveformDataset, DefaultSettingsWaveformDataset, build_waveform_dataset


@dataclass
class SettingsDataset:
    """Class for handling preparation of dataset
    Attributes:
        data_path:              String with path to dataset
        data_file_name:         String with name of dataset file
        use_cell_sort_mode:     Number for building a sub-dataset from original dataset [0: None, 1: Reduced, 2: Type, 3: Group]
        augmentation_do:        Boolean for applying data augmentation (only 1D data)
        augmentation_num:       Number of the samples of each class
        normalization_do:       Boolean for applying data normalization
        normalization_method:   String with applied normalization method ['zeroone', 'minmax', 'norm', 'zscore', 'medianmad', 'meanmad']
        reduce_samples_per_cluster_do:  Boolean for reducing number of samples per class
        reduce_samples_per_cluster_num: Number of reduced samples per class
        exclude_cluster:        List with IDs for excluding cluster/label IDs
    """
    # --- Settings of Datasets
    data_path: str
    data_file_name: str
    use_cell_sort_mode: int
    # --- Data Augmentation
    augmentation_do: bool
    augmentation_num: int
    normalization_do: bool
    normalization_method: str
    reduce_samples_per_cluster_do: bool
    reduce_samples_per_cluster_num: int
    # --- Dataset Preparation
    exclude_cluster: list

    @property
    def get_path2data(self) -> str:
        """Getting the path name to the file"""
        return join(self.get_path2folder, self.data_file_name)

    @property
    def get_path2folder(self) -> str:
        """Getting the path name to the file"""
        if not isabs(self.data_path):
            path = join(self.get_path2folder_project, self.data_path)
        else:
            path = join(self.data_path)
        return abspath(path)

    @property
    def get_path2folder_project(self) -> str:
        """Getting the default path of the Python Project"""
        return get_path_to_project_start()


DefaultSettingsDataset = SettingsDataset(
    data_path='dataset',
    data_file_name='',
    use_cell_sort_mode=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=False,
    normalization_method='minmax',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[]
)


class ControllerDataset:
    _logger: Logger
    _settings: SettingsDataset
    _methods: list
    _index_search: list=['_get_', '_prepare_']
    _path: str

    def __init__(self, settings: SettingsDataset, temp_folder: str='') -> None:
        self._settings = settings
        self._logger = getLogger(__name__)
        self._methods = self._extract_func(self.__class__)
        self._path = join(self._settings.get_path2folder_project, temp_folder)

    def _extract_func(self, class_obj: object) -> list:
        return [method for method in dir(class_obj) if self._index_search[0] in method or self._index_search[1] in method]

    def _extract_methods(self, search_index: str) -> list:
        return [method.split('_')[-1].lower() for method in self._methods if search_index in method]

    def _extract_executive_method(self, search_index: str) -> int:
        used_data_source_idx = -1
        for idx, method in enumerate(self._methods):
            check = method.split(search_index)[-1].lower()
            if self._settings.data_file_name.lower() == check:
                used_data_source_idx = idx
                break
        return used_data_source_idx

    @staticmethod
    def _merge_data(data_train: Tensor, data_test: Tensor) -> np.ndarray:
        return concat(tensors=(data_train, data_test), dim=0).numpy()

    def __download_if_missing(self) -> None:
        idx = self._extract_executive_method(self._index_search[0])
        if idx == -1:
            raise NotImplementedError
        else:
            getattr(self, self._methods[idx])()

    def __process_data(self) -> dict:
        idx = self._extract_executive_method(self._index_search[1])
        if idx == -1:
            raise NotImplementedError
        else:
            return getattr(self, self._methods[idx])()

    def print_overview_datasets(self, do_print: bool=True) -> list:
        """Giving an overview of available datasets on the cloud storage
        :return:            Return a list with dataset names
        """
        oc_handler = OwnCloudDownloader(path2config=self._path)
        list_datasets = self._extract_methods(self._index_search[1])
        list_datasets.extend(oc_handler.get_overview_data(use_dataset=True))
        if do_print:
            self._logger.info("\nAvailable datasets in repository and from remote:")
            self._logger.info("==================================================")
            for idx, file in enumerate(list_datasets):
                self._logger.info(f"\t{idx}: \t{file}")

        oc_handler.close()
        return list_datasets

    def load_dataset(self, do_print: bool=True) -> dict:
        """Loading the dataset from defined data file
        :return:    Dictionary with entries ['data', 'label', 'dict']
        """
        if self._settings.data_file_name.lower() == '':
            self.print_overview_datasets(do_print=do_print)
            raise FileNotFoundError("--- Dataset is not available. Please type-in the data set name into the yaml file ---")
        else:
            makedirs(self._settings.get_path2folder, exist_ok=True)
            self.__download_if_missing()
            return self.__process_data()

    def reconfigure_cluster_with_cell_lib(self, fn, sel_mode_classes: int,
                                          frames_in: np.ndarray, frames_cl: np.ndarray) -> dict:
        """Function for reducing the samples for a given cell bib
        :param fn:                  Class with new resorting dictionaries
        :param sel_mode_classes:    Number of classes to select for each cell bib (0= original, 1= Reduced, 2= Subgroup, 3= Subtype)
        :param frames_in:           Numpy array with spike frames
        :param frames_cl:           Numpy array with cluster label to each spike frame
        :return:                    Dict with ['frame': spike frames, 'cl': new class id, 'dict': new dictionary with names]
        """
        cl_sampler = CellSelector(fn, sel_mode_classes)
        cell_dict = cl_sampler.get_label_list()
        self._logger.info(f"... Cluster types before reconfiguration: {np.unique(frames_cl)}")

        cluster_new, data_new = cl_sampler.transform_data_into_new(frames_cl, frames_in)
        self._logger.info(f"... Cluster types after reconfiguration: {np.unique(cluster_new)}")
        return {'frame': data_new, 'cl': cluster_new, 'dict': cell_dict}

    def __pipeline_for_torchvision_datasets(self, picture: np.ndarray, label: np.ndarray) -> dict:
        # --- Normalization
        if self._settings.normalization_do:
            picture = picture / 255.0
            self._logger.info("... do data normalization on input")

        # --- Exclusion of selected clusters
        if len(self._settings.exclude_cluster):
            for i, id in enumerate(self._settings.exclude_cluster):
                selX = np.where(label != id)
                picture = picture[selX[0], :]
                label = label[selX]
            self._logger.info(f"... class reduction done to {np.unique(label).size} classes")

        # --- Using cell library
        if self._settings.use_cell_sort_mode:
            raise NotImplementedError("No cell library for this case is available - Please disable flag!")

        # --- Data Augmentation
        if self._settings.augmentation_do:
            raise NotImplementedError("No augmentation method is implemented - Please disable flag!")

        if self._settings.reduce_samples_per_cluster_do:
            raise NotImplementedError(f"No reducing samples technique is implemented - Please disable flag!")
        return {'data': picture, 'label': label}

    def __get_mnist(self) -> None:
        do_download = not exists(self._settings.get_path2data)
        datasets.MNIST(self._settings.get_path2folder, train=True, download=do_download)
        datasets.MNIST(self._settings.get_path2folder, train=False, download=do_download)

    def __prepare_mnist(self) -> dict:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        data_train = datasets.MNIST(self._settings.get_path2folder, train=True, download=False, transform=transform)
        data_valid = datasets.MNIST(self._settings.get_path2folder, train=False, download=False,
                                    transform=transform)
        data_process = self.__pipeline_for_torchvision_datasets(
            picture=self._merge_data(data_train.data, data_valid.data),
            label=self._merge_data(data_train.targets, data_valid.targets)
        )
        return {'data': data_process['data'], 'label': data_process['label'], 'dict': data_train.classes}

    def __get_fashion(self) -> None:
        do_download = not exists(self._settings.get_path2data)
        datasets.FashionMNIST(self._settings.get_path2folder, train=True, download=do_download)
        datasets.FashionMNIST(self._settings.get_path2folder, train=False, download=do_download)

    def __prepare_fashion(self) -> dict:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        data_train = datasets.FashionMNIST(self._settings.get_path2folder, train=True, download=False,
                                           transform=transform)
        data_valid = datasets.FashionMNIST(self._settings.get_path2folder, train=False, download=False,
                                           transform=transform)
        data_process = self.__pipeline_for_torchvision_datasets(
            picture=self._merge_data(data_train.data, data_valid.data),
            label=self._merge_data(data_train.targets, data_valid.targets)
        )
        return {'data': data_process['data'], 'label': data_process['label'], 'dict': data_train.classes}

    def __get_cifar10(self) -> None:
        do_download = not exists(self._settings.get_path2data)
        datasets.CIFAR10(self._settings.get_path2folder, train=True, download=do_download)
        datasets.CIFAR10(self._settings.get_path2folder, train=False, download=do_download)

    def __prepare_cifar10(self) -> dict:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        data_train = datasets.CIFAR10(self._settings.get_path2folder, train=True, download=False,
                                      transform=transform)
        data_valid = datasets.CIFAR10(self._settings.get_path2folder, train=False, download=False,
                                      transform=transform)
        data_process = self.__pipeline_for_torchvision_datasets(
            picture=self._merge_data(from_numpy(data_train.data), from_numpy(data_valid.data)),
            label=self._merge_data(Tensor(data_train.targets), Tensor(data_valid.targets))
        )
        return {'data': data_process['data'], 'label': data_process['label'], 'dict': data_train.classes}

    def __get_cifar100(self) -> None:
        do_download = not exists(self._settings.get_path2data)
        datasets.CIFAR100(self._settings.get_path2folder, train=True, download=do_download)
        datasets.CIFAR100(self._settings.get_path2folder, train=False, download=do_download)

    def __prepare_cifar100(self) -> dict:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        data_train = datasets.CIFAR100(self._settings.get_path2folder, train=True, download=False,
                                       transform=transform)
        data_valid = datasets.CIFAR100(self._settings.get_path2folder, train=False, download=False,
                                       transform=transform)
        data_process = self.__pipeline_for_torchvision_datasets(
            picture=self._merge_data(from_numpy(data_train.data), from_numpy(data_valid.data)),
            label=self._merge_data(Tensor(data_train.targets), Tensor(data_valid.targets))
        )
        return {'data': data_process['data'], 'label': data_process['label'], 'dict': data_train.classes}

    def __get_waveforms(self) -> SettingsWaveformDataset:
        return YamlHandler(
            template=DefaultSettingsWaveformDataset,
            path='config',
            file_name='Config_Waveform_DatasetBuild'
        ).get_class(SettingsWaveformDataset)

    def __prepare_waveforms(self) -> dict:
        return build_waveform_dataset(
            settings_data=self.__get_waveforms()
        )
