from os import remove, makedirs, cpu_count
from os.path import join
import platform
import subprocess
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from logging import getLogger, Logger
from random import seed
from shutil import rmtree
from glob import glob
from datetime import datetime
from torch import (device, cuda, backends, randn, cat, Tensor, is_tensor, zeros, unique, argwhere, float32,
                   Generator, manual_seed, use_deterministic_algorithms, nn, optim)
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from sklearn.model_selection import KFold

from denspp.offline import get_path_to_project, check_elem_unique
from denspp.offline.dnn.model_library import ModelLibrary
from denspp.offline.dnn.data_config import SettingsDataset
from denspp.offline.structure_builder import init_dnn_folder
from denspp.offline.data_format.yaml import YamlHandler


@dataclass
class SettingsPytorch:
    """Class for handling the PyTorch training/inference pipeline
    Attributes:
        model_name:         String with the model name
        patience:           Integer value with number of epochs before early stopping
        optimizer:          String with PyTorch optimizer name
        loss:               String with method name for the loss function
        deterministic_do:   Boolean if deterministic training should be done
        deterministic_seed: Integer with the seed for deterministic training
        num_kfold:          Integer value with applying k-fold cross validation
        num_epochs:         Integer value with number of epochs
        batch_size:         Integer value with batch size
        data_split_ratio:   Float value for splitting the input dataset between training and validation
        data_do_shuffle:    Boolean if data should be shuffled before training
        custom_metrics:     List with string of custom metrics to calculate during training
    """
    model_name: str
    patience: int
    optimizer: str
    loss: str
    deterministic_do: bool
    deterministic_seed: int
    num_kfold: int
    num_epochs: int
    batch_size: int
    data_split_ratio: float
    data_do_shuffle: bool
    custom_metrics: list

    @staticmethod
    def get_model_overview(print_overview: bool=False, index: str='') -> list:
        """Function for getting an overview of existing models inside library"""
        models_bib = ModelLibrary().get_registry()
        return models_bib.get_library_overview(index, do_print=print_overview)

    def get_loss_func(self) -> Any:
        """Getting the loss function"""
        match self.loss:
            case 'L1':
                loss_func = nn.L1Loss
            case 'MSE':
                loss_func = nn.MSELoss()
            case 'Cross Entropy':
                loss_func = nn.CrossEntropyLoss()
            case 'Cosine Similarity':
                loss_func = nn.CosineSimilarity()
            case _:
                raise NotImplementedError("Loss function unknown! - Please implement or check!")
        return loss_func

    def load_optimizer(self, model, learn_rate: float=0.1) -> Any:
        """Loading the optimizer function
        :param model:       PyTorch Sequential of the model with pre-defined configuration
        :param learn_rate:  Learning rate of the optimizer
        :return:            PyTorch Optimizer
        """
        match self.optimizer:
            case 'Adam':
                optim_func = optim.Adam(model.parameters())
            case 'SGD':
                optim_func = optim.SGD(model.parameters(), lr=learn_rate)
            case _:
                raise NotImplementedError("Optimizer function unknown! - Please implement or check!")
        return optim_func

    def get_model(self, *args, **kwargs):
        """Function for loading the model to train"""
        models_bib = ModelLibrary().get_registry()
        if not self.model_name:
            models_bib.get_library_overview(do_print=True)
            raise AttributeError("Please select one model above and type-in the name into yaml file")
        else:
            if models_bib.check_module_available(self.model_name):
                used_model = deepcopy(models_bib.build(self.model_name, *args, **kwargs))
                return used_model
            else:
                models_bib.get_library_overview(do_print=True)
                raise AttributeError(f"Model is not available - Please check again!")


DefaultSettingsTrainMSE = SettingsPytorch(
    model_name='',
    patience=20,
    optimizer='Adam',
    loss='MSE',
    deterministic_do=False,
    deterministic_seed=42,
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.2,
    custom_metrics=[]
)
DefaultSettingsTrainCE = SettingsPytorch(
    model_name='',
    patience=20,
    optimizer='Adam',
    loss='Cross Entropy',
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.2,
    deterministic_do=False,
    deterministic_seed=42,
    custom_metrics=[]
)


class PyTorchHandler:
    _deterministic_generator: Generator
    _used_hw_dev: device
    _used_hw_num: int
    _train_loader: list
    _valid_loader: list
    _selected_samples: dict
    _cell_classes: list
    _metric_methods: dict
    _ptq_do_validation: bool = False
    _ptq_level: list = [12, 8]
    _logger: Logger

    def __init__(self, config_train: SettingsPytorch, config_dataset: SettingsDataset, do_train: bool=True) -> None:
        """Class for Handling Training of Deep Neural Networks in PyTorch
        Args:
            config_train:   Configuration settings for the PyTorch Training
            config_dataset: Configuration settings for dataset handling
            do_train:       Mention if training should be used (default = True)
        Returns:
            None
        """
        init_dnn_folder()
        self._logger = getLogger(__name__)
        # --- Preparing Neural Network
        self._model = None
        self._loss_fn = None
        self._optimizer = None
        # --- Preparing options
        self._config_available = False
        self._kfold_do = False
        self._shuffle_do = config_train.data_do_shuffle
        self._kfold_run = 0
        # --- Saving options
        self._settings_train: SettingsPytorch = config_train
        self._settings_data: SettingsDataset = config_dataset
        self._index_folder = 'train' if do_train else 'inference'
        self._model_addon = str()
        # --- Logging paths for saving
        self.__check_start_folder()
        self._path2save = str()
        self._path2log = str()
        self._path2temp = str()
        self._path2config = str()

    @staticmethod
    def _get_cpu_name_windows() -> str:
        return platform.processor()

    @staticmethod
    def _get_cpu_name_mac() -> str:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
        return result.stdout.strip()

    @staticmethod
    def _get_cpu_name_linux():
        result = subprocess.run(['cat', '/proc/cpuinfo'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1).strip()

    def _get_cpu_name(self) -> str:
        match platform.system().lower():
            case 'windows':
                return self._get_cpu_name_windows()
            case 'linux':
                return self._get_cpu_name_linux()
            case 'darwin':
                return self._get_cpu_name_mac()
            case _:
                return ''

    def __check_start_folder(self, new_folder: str='runs'):
        """Checking for starting folder to generate"""
        self._path2run = get_path_to_project(new_folder)
        makedirs(self._path2run, exist_ok=True)

    def __setup_device(self) -> None:
        if cuda.is_available():
            # Using GPU
            used_hw_gpu = cuda.get_device_name()
            self._used_hw_dev = device("cuda")
            self._used_hw_num = cuda.device_count()
            device0 = used_hw_gpu
            cuda.empty_cache()
        elif backends.mps.is_available() and backends.mps.is_built() and platform.system().lower() == "darwin":
            # Using Apple M1 Chip
            self._used_hw_dev = device("mps")
            self._used_hw_num = cuda.device_count()
            device0 = self._get_cpu_name()
        else:
            # Using normal CPU
            self._used_hw_dev = device("cpu")
            self._used_hw_num = cpu_count()
            device0 = self._get_cpu_name()
        self._logger.debug(f"\nUsing PyTorch with {device0} on {platform.system()}")

    def _init_train(self, path2save: str='', addon: str='') -> None:
        """Do init of class for training"""
        if not path2save:
            folder_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self._index_folder}_{self._model.__class__.__name__}'
            self._path2save = join(self._path2run, folder_name)
        else:
            self._path2save = path2save
        self._path2temp = join(self._path2save, f'temp')

        # --- Generate folders
        makedirs(self._path2run, exist_ok=True)
        makedirs(self._path2save, exist_ok=True)
        makedirs(self._path2temp, exist_ok=True)

        # --- Transfer model to hardware
        self._model.to(device=self._used_hw_dev)

        # --- Copy settings to YAML file
        YamlHandler(
            template=self._settings_data,
            path=self._path2save,
            file_name='Config_Dataset'
        )
        YamlHandler(
            template=self._settings_train,
            path=self._path2save,
            file_name=f'Config_Training{addon}'
        )

    def __deterministic_training_preparation(self) -> None:
        """Preparing the CUDA hardware for deterministic training"""
        if self._settings_train.deterministic_do:
            np.random.seed(self._settings_train.deterministic_seed)
            manual_seed(self._settings_train.deterministic_seed)
            if cuda.is_available():
                cuda.manual_seed_all(self._settings_train.deterministic_seed)
            seed(self._settings_train.deterministic_seed)
            backends.cudnn.deterministic = True

            use_deterministic_algorithms(True)
            self._logger.info(f"=== DL Training with Deterministic @seed: {self._settings_train.deterministic_seed} ===")
        else:
            use_deterministic_algorithms(False)
            self._logger.info(f"=== Normal DL Training ===")

    def __deterministic_get_dataloader_params(self) -> dict:
        """Getting the parameters for preparing the Training and Validation DataLoader for Deterministic Training"""
        if self._settings_train.deterministic_do:
            self._deterministic_generator = Generator()
            self._deterministic_generator.manual_seed(self._settings_train.deterministic_seed)
            worker_init_fn = lambda worker_id: np.random.seed(self._settings_train.deterministic_seed)
            return {'worker_init_fn': worker_init_fn, 'generator': self._deterministic_generator}
        else:
            return {}

    def _prepare_dataset_for_training(self, data_set, num_workers: int=0) -> None:
        """Loading data for training and validation in DataLoader format into class
        Args:
            data_set:        Dataclass DatasetFromFil loaded from file
            num_workers:    Number of workers for calculation [Default: 0 --> single core]
        Return:
            None
        """
        self.__setup_device()
        self._kfold_do = True if self._settings_train.num_kfold > 1 else False
        self._model_addon = data_set.get_topology_type
        self._cell_classes = data_set.get_dictionary
        params_deterministic = self.__deterministic_get_dataloader_params()

        # --- Preparing datasets
        out_train = list()
        out_valid = list()
        if self._kfold_do:
            kfold = KFold(n_splits=self._settings_train.num_kfold,
                          shuffle=self._shuffle_do and not self._settings_train.deterministic_do)
            for idx_train, idx_valid in kfold.split(np.arange(len(data_set))):
                subsamps_train = SubsetRandomSampler(idx_train)
                subsamps_valid = SubsetRandomSampler(idx_valid)
                out_train.append(DataLoader(data_set,
                                            batch_size=self._settings_train.batch_size,
                                            sampler=subsamps_train,
                                            **params_deterministic))
                out_valid.append(DataLoader(data_set,
                                            batch_size=self._settings_train.batch_size,
                                            sampler=subsamps_valid,
                                            **params_deterministic))
        else:
            idx = np.arange(len(data_set))
            if self._shuffle_do and not self._settings_train.deterministic_do:
                np.random.shuffle(idx)
            split_pos = int(len(data_set) * (1 - self._settings_train.data_split_ratio))
            idx_train = idx[0:split_pos]
            idx_valid = idx[split_pos:]
            subsamps_train = SubsetRandomSampler(idx_train)
            subsamps_valid = SubsetRandomSampler(idx_valid)
            out_train.append(DataLoader(data_set,
                                        batch_size=self._settings_train.batch_size,
                                        sampler=subsamps_train,
                                        **params_deterministic))
            out_valid.append(DataLoader(data_set,
                                        batch_size=self._settings_train.batch_size,
                                        sampler=subsamps_valid,
                                        **params_deterministic))

        # --- CUDA support for dataset
        if cuda.is_available():
            for idx, dataset in enumerate(out_train):
                out_train[idx].pin_memory = True
                out_train[idx].pin_memory_device = self._used_hw_dev.type
                out_train[idx].num_workers = num_workers

                out_valid[idx].pin_memory = True
                out_valid[idx].pin_memory_device = self._used_hw_dev.type
                out_valid[idx].num_workers = num_workers

        # --- Output: Data
        self._train_loader = out_train
        self._valid_loader = out_valid

    def get_saving_path(self) -> str:
        """Getting the path for saving files in aim folder"""
        return self._path2save

    def get_best_model(self, type_model: str) -> list:
        """Getting the path to the best trained model"""
        return glob(join(self._path2save, f'*{type_model}*.pt'))

    def load_model(self, model, learn_rate: float=0.1) -> None:
        """Loading optimizer, loss_fn into class
        Args:
            model:          PyTorch Neural Network for Training / Inference
            learn_rate:     Learning rate used for SGD optimier
        Returns:
            None
        """
        self._model = model
        self._optimizer = self._settings_train.load_optimizer(model, learn_rate=learn_rate)
        self._loss_fn = self._settings_train.get_loss_func()

        # --- Init. hardware for deterministic training
        if self._settings_train.deterministic_do:
            self.__deterministic_training_preparation()

        # --- Print model
        self._logger.info("\nPrint summary of model")
        self._logger.info(str(summary(self._model, input_size=self._model.model_shape)))
        self._logger.info("\n\n")

    def _save_train_results(self, last_metric_train: float | np.ndarray,
                            last_metric_valid: float | np.ndarray, loss_type: str='Loss') -> None:
        """Writing some training metrics into txt-file"""
        if self._config_available:
            with open(self._path2config, 'a') as txt_handler:
                txt_handler.write(f'\n--- Metrics of last epoch in fold #{self._kfold_run} ---')
                txt_handler.write(f'\nTraining {loss_type} = {last_metric_train}')
                txt_handler.write(f'\nValidation {loss_type} = {last_metric_valid}\n')

    def _end_training_routine(self, timestamp_start: datetime, do_delete_temps: bool=True) -> None:
        """Doing the last step of training routine"""
        timestamp_end = datetime.now()
        timestamp_string = timestamp_end.strftime('%H:%M:%S')
        diff_time = timestamp_end - timestamp_start
        diff_string = diff_time
        self._logger.info(f'\nTraining ends on: {timestamp_string}')
        self._logger.info(f'Training runs: {diff_string}')

        # Delete init model
        init_model = glob(join(self._path2save, '*_reset.pt'))
        for file in init_model:
            remove(file)

        # Delete log folders
        if do_delete_temps:
            folder_logs = glob(join(self._path2save, 'temp*'))
            for folder in folder_logs:
                rmtree(folder, ignore_errors=True)

    def __get_data_points(self, only_getting_labels: bool=False, use_train_dataloader: bool=False) -> dict:
        """Getting data from DataLoader for Plotting Results
        Args:
            only_getting_labels:    Option for taking only labels
            use_train_dataloader:   Mode for selecting datatype (True=Training, False=Validation)
        Returns:
              Dict with data for plotting
        """
        used_dataset = self._train_loader[-1] if use_train_dataloader else self._valid_loader[-1]

        # --- Getting the keys
        keys = list()
        for data in used_dataset:
            keys = list(data.keys())
            break

        if only_getting_labels:
            keys.pop(0)

        # --- Extracting data
        data_extract = [randn(32, 1) for _ in keys]
        first_run = True
        for data in used_dataset:
            for idx, key in enumerate(keys):
                if first_run:
                    data_extract[idx] = data[key]
                else:
                    data_extract[idx] = cat((data_extract[idx], data[key]), dim=0)
            first_run = False

        # --- Prepare output
        mdict = dict()
        for idx, data in enumerate(data_extract):
            mdict.update({keys[idx]: data.numpy()})
        return mdict

    def _getting_data_for_plotting(self, valid_input: np.ndarray, valid_label: np.ndarray, results=None, addon: str='') -> dict:
        """Getting the raw data for plotting results
        :param valid_input: Numpy array with input data for training validation
        :param valid_label: Numpy array with labels for training validation
        :param results:     Option for plotting results
        """
        if results is None:
            results = dict()

        self._logger.info(f"... preparing results for plot generation")
        data_train = self.__get_data_points(
            only_getting_labels=True,
            use_train_dataloader=True
        )

        output = {
            'settings': self._settings_train,
            'date': datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
            'train_clus': data_train['class'] if addon == 'ae' else data_train['out'],
            'cl_dict': self._cell_classes,
            'input': valid_input,
            'valid_clus': valid_label
        }
        output.update(results)

        data2save = join(self.get_saving_path(), f'results_{addon}.npy')
        self._logger.debug(f"... saving results: {data2save}")
        np.save(data2save, output)
        return output

    def _determine_epoch_metrics(self, do_metrics: str):
        """Determination of additional metrics during training
        Args:
            do_metrics:     String with index for calculating epoch metric
        Return:
            Function for metric calculation
        """
        func = Tensor
        for metric_avai, func in self._metric_methods.items():
            if metric_avai == do_metrics:
                break
        return func

    def _separate_classes_from_label(self, pred: Tensor, true: Tensor, label: str, *args) -> tuple[Tensor, Tensor]:
        """Separating the classes for further metric processing
        Args:
            pred:           Torch Tensor from prediction
            true:           Torch Tensor from labeled dataset (ground-truth)
            key:            String with processing metric
            func:           Function for metric calculation
        Return:
            Calculated metric results in Tensor array and total samples of each class
        """
        if args or not "cl" in label:
            metric_out = zeros((len(self._cell_classes),), dtype=float32)
        else:
            metric_out = [zeros((1,)) for _ in self._cell_classes]

        length_out = zeros((len(self._cell_classes),), dtype=float32)
        for idx, id in enumerate(unique(true)):
            xpos = argwhere(true == id).flatten()
            length_out[idx] = len(xpos)
            if args:
                metric_out[idx] += args[0](pred[xpos], true[xpos])
            else:
                metric_out[idx] = pred[xpos]
        return metric_out, length_out

    @staticmethod
    def _converting_tensor_to_numpy(metric_used: dict) -> dict:
        """Converting tensor array to numpy for later processing
        :param metric_used: Dictionary of used metric
        :return:            Dictionary with calculated metrics
        """
        metric_save = deepcopy(metric_used)
        for key0, data0 in metric_used.items():
            for key1, data1 in data0.items():
                for idx2, data2 in enumerate(data1):
                    if isinstance(data2, list):
                        for idx3, data3 in enumerate(data2):
                            if is_tensor(data3):
                                metric_save[key0][key1][idx2][idx3] = data3.cpu().detach().numpy()
                    else:
                        if is_tensor(data2):
                            metric_save[key0][key1][idx2] = data2.cpu().detach().numpy()
        return metric_save

    @property
    def get_epoch_metric_custom_methods(self) -> list:
        """Getting an overview of available methods for custom-written metric calculation in each epoch during training
        :return:    List with metrics name to call
        """
        return [key for key in self._metric_methods.keys()]

    @property
    def get_number_parameters_from_model(self) -> int:
        """Getting the number of used parameters of used DNN model"""
        return int(sum(p.numel() for p in self._model.parameters()))

    def define_ptq_level(self, total_bitwidth: int, frac_bitwidth: int) -> None:
        """Function for defining the post-training quantization level of the model
        :param total_bitwidth: Total bitwidth of the model
        :param frac_bitwidth: Fraction of bitwidth used for quantization
        :return: None
        """
        if frac_bitwidth < 0 or frac_bitwidth > total_bitwidth:
            raise ValueError(f"Fraction of bitwidth must be between 0 and {total_bitwidth}")
        if total_bitwidth < 0:
            raise ValueError(f"Total bitwidth must be greater than 0")

        self._ptq_level = [total_bitwidth, frac_bitwidth]
