import numpy as np
from pathlib import Path
from logging import getLogger, Logger
from dataclasses import dataclass
from denspp.offline import get_path_to_project, check_elem_unique
from denspp.offline.data_call.remote_handler import RemoteDownloader


@dataclass(frozen=True)
class DatasetFromFile:
    """Dataclass with data, labels and dict loaded externally
    Attributes:
        data:   Numpy array with dataset content, shape = [num. samples, dimension]
        label:  Numpy array with labels, shape = [num. samples]
        dict:   List with names for each class/label
        mean:   Numpy array with mean values, shape = [num. samples, dimension]
    """
    data: np.ndarray
    label: np.ndarray
    dict: list
    mean: np.ndarray


@dataclass
class SettingsDataset:
    """Class for handling preparation of dataset
    Attributes:
        data_path:              String with path to dataset
        data_type:              String with name of unique key to identify dataset to load [e.g. Waveform, MNIST, ...]
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
    data_type: str
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
    def get_path2folder(self) -> Path:
        """Getting the path name to the file"""
        if not Path(self.data_path).is_absolute():
            path = self.get_path2folder_project / self.data_path
        else:
            path = Path(self.data_path)
        return path.absolute()

    @property
    def get_path2folder_project(self) -> Path:
        """Getting the default path of the Python Project"""
        return Path(get_path_to_project())


DefaultSettingsDataset = SettingsDataset(
    data_path='dataset',
    data_type='',
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
    _path: Path

    def __init__(self, settings: SettingsDataset, temp_folder: str='') -> None:
        self._settings = settings
        self._logger = getLogger(__name__)
        self._methods = self._extract_func(self.__class__)
        self._path = self._settings.get_path2folder_project / temp_folder

    @property
    def get_overview_methods(self) -> list:
        """Returning a list with string of all available dataset methods"""
        return self._methods

    @property
    def get_path2folder(self) -> Path:
        """Returning the absolute path to the folder"""
        return self._path.absolute()

    def _extract_func(self, class_obj: object) -> list:
        return [method for method in dir(class_obj) if self._index_search[0] in method or self._index_search[1] in method]

    def _extract_methods(self, search_index: str) -> list:
        return [method.split('_')[-1].lower() for method in self._methods if search_index in method]

    def _extract_executive_method(self, search_index: str) -> int:
        used_data_source_idx = -1
        for idx, method in enumerate(self._methods):
            check = method.split(search_index)[-1].lower()
            if self._settings.data_type.lower() == check:
                used_data_source_idx = idx
                break
        return used_data_source_idx

    def __download_if_missing(self) -> None:
        idx = self._extract_executive_method(self._index_search[0])
        if idx == -1:
            raise NotImplementedError
        else:
            getattr(self, self._methods[idx])()

    def __process_data(self) -> DatasetFromFile:
        idx = self._extract_executive_method(self._index_search[1])
        if idx == -1:
            raise NotImplementedError
        else:
            return getattr(self, self._methods[idx])()

    def print_overview_datasets(self, do_print: bool=True) -> list:
        """Giving an overview of available datasets on the cloud storage
        :return:            Return a list with dataset names
        """
        oc_handler = RemoteDownloader(path2config=str(self._path))
        list_datasets = self._extract_methods(self._index_search[1])
        list_datasets.extend(oc_handler.get_overview_data(use_dataset=True))
        if do_print:
            self._logger.info("\nAvailable datasets in repository and from remote:")
            self._logger.info("==================================================")
            for idx, file in enumerate(list_datasets):
                self._logger.info(f"\t{idx}: \t{file}")

        oc_handler.close()
        return list_datasets

    def print_dataset_properties(self, data: DatasetFromFile) -> None:
        """Printing the properties of the loaded dataset
        :param data:    Dataclas DatasetFromFile loaded externally
        :return:        None
        """
        check = np.unique(data.label, return_counts=True)
        self._logger.info(f"... for training are {data.data.shape[0]} frames with each "
                          f"({data.data.shape[1]}, {data.data.shape[2]}) points available")
        self._logger.info(f"... used data points for training: "
                          f"in total {check[0].size} classes with {np.sum(check[1])} samples")
        for idx, id in enumerate(check[0]):
            addon = f'' if len(data.dict) == 0 else f' ({data.dict[idx]})'
            self._logger.info(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    def load_dataset(self, do_print: bool=True) -> DatasetFromFile:
        """Loading the dataset from defined data file
        :return:    Dataclass DatasetFromFile with attributes ['data', 'label', 'dict', 'mean']
        """
        if self._settings.data_type.lower() == '':
            self.print_overview_datasets(do_print=do_print)
            raise AttributeError("--- Dataset is not available. Please type-in the data set name into the yaml file ---")
        else:
            self._settings.get_path2folder.mkdir(parents=True, exist_ok=True)
            self.__download_if_missing()
            return self.__process_data()

    def _download_file(self, dataset_name: str) -> None:
        # TODO: Error - Files will always be downloaded
        # TODO: Definition of get_path2data is wrong
        if not self._settings.get_path2folder.exists():
            oc_handler = RemoteDownloader(str(self._path))
            oc_handler.download_file(
                use_dataset=True,
                file_name=dataset_name,
                destination_download=str(self._settings.get_path2folder / dataset_name)
            )
            oc_handler.close()


@dataclass(frozen=True)
class TransformLabels:
    """
    Dataclass for transforming true and predicted labels into new scheme
    Attributes:
        true:    Numpy array with true labels
        pred:    Numpy array with predicted labels
    """
    true: np.ndarray
    pred: np.ndarray


def logic_combination(labels_in: TransformLabels, translate_list: list) -> TransformLabels:
    """Combination of logic for Reducing Label Classes
    :param labels_in:       Dataclass with labels for true and predicted case
    :param translate_list:  List with label ids to combine (e.g. [[1, 2], [0, 3]] -> [0, 1])
    :returns:               Transformed new dataclass
    """
    assert labels_in.true.shape == labels_in.pred.shape, "Shape of labels are not equal"
    assert len(translate_list), "List with new translation is empty"
    assert check_elem_unique(translate_list), "Not all key elements in sublists are unique"

    true_labels_new = np.zeros_like(labels_in.true, dtype=np.uint8)
    pred_labels_new = np.zeros_like(labels_in.pred, dtype=np.uint8)

    for idx, cluster in enumerate(translate_list):
        for id in cluster:
            pos = np.argwhere(labels_in.true == id).flatten()
            true_labels_new[pos] = idx
            pos = np.argwhere(labels_in.pred == id).flatten()
            pred_labels_new[pos] = idx
    return TransformLabels(
        true=true_labels_new,
        pred=pred_labels_new
    )
