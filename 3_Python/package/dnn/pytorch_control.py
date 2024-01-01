import dataclasses
import platform
import shutil

import cpuinfo
import numpy as np
from typing import Any
from os import mkdir, remove
from os.path import exists, join
from shutil import rmtree
from glob import glob
from datetime import datetime
from torch import optim, device, cuda, backends
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from sklearn.model_selection import KFold


@dataclasses.dataclass(frozen=True)
class Config_PyTorch:
    """Class for handling the PyTorch training/inference routing"""
    # --- Settings of Models/Training
    model: Any
    loss_fn: Any
    optimizer: str
    loss: str
    num_kfold: int
    num_epochs: int
    batch_size: int
    data_split_ratio: float
    data_do_shuffle: bool

    def get_topology(self) -> str:
        """Getting the model name defined in models"""
        return self.model.out_modeltyp

    def load_optimizer(self, learn_rate=0.1) -> Any:
        """Loading the optimizer function"""
        if self.optimizer == 'Adam':
            return self.__set_optimizer_adam()
        elif self.optimizer == 'SGD':
            return self.__set_optimizer_sgd(learn_rate=learn_rate)
        else:
            return -1

    def __set_optimizer_adam(self):
        """Using the Adam Optimizer"""
        return optim.Adam(self.model.parameters())

    def __set_optimizer_sgd(self, learn_rate=0.1):
        """Using the SGD as Optimizer"""
        return optim.SGD(self.model.parameters(), lr=learn_rate)


@dataclasses.dataclass(frozen=True)
class Config_Dataset:
    """Class for handling preparation of dataset"""
    # --- Settings of Datasets
    data_path: str
    data_file_name: str
    # --- Data Augmentation
    data_do_augmentation: bool
    data_num_augmentation: int
    data_do_normalization: bool
    data_do_addnoise_cluster: bool
    data_do_reduce_samples_per_cluster: bool
    data_num_samples_per_cluster: int
    # --- Dataset Preparation
    data_exclude_cluster: list
    data_sel_pos: list

    def get_path2data(self) -> str:
        """Getting the path name to the file"""
        return join(self.data_path, self.data_file_name)




class training_pytorch:
    """Class for Handling Training of Deep Neural Networks in PyTorch
    Args:
        config_train: Configuration settings for the PyTorch Training
        do_train: Mention if training should be used (default = True)
    """
    used_hw_dev: device
    used_hw_cpu: str
    used_hw_gpu: str
    used_hw_num: int
    train_loader: list
    valid_loader: list
    cell_classes: list

    def __init__(self, config_train: Config_PyTorch, config_dataset=Config_Dataset, do_train=True) -> None:
        self.os_type = platform.system()
        self._writer = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.data_set = None

        # --- Preparing options
        self.config_available = False
        self._do_kfold = False
        self._do_shuffle = config_train.data_do_shuffle
        self._run_kfold = 0
        self._samples_train = list()
        self._samples_valid = list()

        # --- Saving options
        self.settings = config_train
        self.data_settings = config_dataset
        self._index_folder = 'train' if do_train else 'inference'
        self._aitype = config_train.model.out_modeltyp
        self._model_name = config_train.model.out_modelname
        self._model_addon = str()
        self._path2run = 'runs'
        self._path2save = str()
        self._path2log = str()
        self._path2temp = str()
        self._path2config = str()

    def __setup_device(self, use_only_cpu=True) -> None:
        """Setup PyTorch for Training

        Args:
            use_only_cpu: Set if in the training oly the CPU should be used
            """
        # Using GPU
        if cuda.is_available() and not use_only_cpu:
            self.used_hw_gpu = cuda.get_device_name()
            self.used_hw_cpu = (f"{cpuinfo.get_cpu_info()['brand_raw']} "
                       f"(@ {1e-9 * cpuinfo.get_cpu_info()['hz_actual'][0]:.3f} GHz)")
            self.used_hw_dev = device("cuda")
            self.used_hw_num = cuda.device_count()
            device0 = self.used_hw_gpu
        # Using Apple M1 Chip
        elif backends.mps.is_available() and backends.mps.is_built() and self.os_type == "Darwin" and not use_only_cpu:
            self.used_hw_cpu = "MP1"
            self.used_hw_gpu = 'None'
            self.used_hw_num = cuda.device_count()
            self.used_hw_dev = device("mps")
            self.used_hw_num = 1
            device0 = self.used_hw_cpu
        # Using normal CPU
        else:
            self.used_hw_cpu = (f"{cpuinfo.get_cpu_info()['brand_raw']} "
                       f"(@ {1e-9 * cpuinfo.get_cpu_info()['hz_actual'][0]:.3f} GHz)")
            self.used_hw_gpu = 'None'
            self.used_hw_dev = device("cpu")
            self.used_hw_num = cpuinfo.get_cpu_info()['count']
            device0 = self.used_hw_cpu

        print(f"... using PyTorch with {device0} device on {self.os_type}")

    def _check_user_config(self) -> None:
        if not exists("settings_ai"):
            shutil.copy()

    def _init_train(self, path2save='') -> None:
        """Do init of class for training"""
        if not exists(self._path2run):
            mkdir(self._path2run)

        if not path2save:
            folder_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self._index_folder}_{self._model_name}'
            self._path2save = join(self._path2run, folder_name)
        else:
            self._path2save = path2save

        if not exists(self._path2save):
            mkdir(self._path2save)

        self._path2temp = join(self._path2save, f'temp')
        mkdir(self._path2temp)

        # --- Sending everything to device
        self.model.to(device=self.used_hw_dev)
        # self.loss_fn.to(devive=self.used_hw_dev)

    def _init_writer(self) -> None:
        """Do init of writer"""
        self._path2log = join(self._path2save, f'logs')
        self._writer = SummaryWriter(self._path2log, comment=f"event_log_kfold{self._run_kfold:03d}")

    def load_data(self, data_set, use_not_cpu=False) -> None:
        self.__setup_device()
        """Loading data for training and validation in DataLoader format into class"""
        self._do_kfold = True if self.settings.num_kfold > 1 else False
        self._model_addon = data_set.data_type
        self.cell_classes = data_set.frame_dict if data_set.cluster_name_available else []

        # --- Preparing datasets
        out_train = list()
        out_valid = list()
        if self._do_kfold:
            kfold = KFold(n_splits=self.settings.num_kfold, shuffle=self._do_shuffle)
            for idx_train, idx_valid in kfold.split(np.arange(len(data_set))):
                subsamps_train = SubsetRandomSampler(idx_train)
                subsamps_valid = SubsetRandomSampler(idx_valid)
                out_train.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_train,
                                            pin_memory=use_not_cpu, pin_memory_device=self.used_hw_dev.type))
                out_valid.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_valid,
                                            pin_memory=use_not_cpu, pin_memory_device=self.used_hw_dev.type))
                self._samples_train.append(subsamps_train.indices.size)
                self._samples_valid.append(subsamps_valid.indices.size)
        else:
            idx = np.arange(len(data_set))
            if self._do_shuffle:
                np.random.shuffle(idx)
            split_pos = int(len(data_set) * (1 - self.settings.data_split_ratio))
            idx_train = idx[0:split_pos]
            idx_valid = idx[split_pos:]
            subsamps_train = SubsetRandomSampler(idx_train)
            subsamps_valid = SubsetRandomSampler(idx_valid)
            out_train.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_train,
                                        pin_memory=use_not_cpu, pin_memory_device=self.used_hw_dev.type))
            out_valid.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_valid,
                                        pin_memory=use_not_cpu, pin_memory_device=self.used_hw_dev.type))
            self._samples_train.append(subsamps_train.indices.size)
            self._samples_valid.append(subsamps_valid.indices.size)

        # --- Output
        self.train_loader = out_train
        self.valid_loader = out_valid

    def load_model(self, learn_rate=0.1, print_model=True) -> None:
        """Loading optimizer, loss_fn into class"""
        self.model = self.settings.model
        self.optimizer = self.settings.load_optimizer(learn_rate=learn_rate)
        self.loss_fn = self.settings.loss_fn
        if print_model:
            summary(self.model, input_size=self.model.model_shape)

    def _save_config_txt(self, addon='') -> None:
        """Writing the content of the configuration class in *.txt-file"""
        self._path2config = join(self._path2save, f'config{addon}.txt')
        self.config_available = True

        with open(self._path2config, 'w') as txt_handler:
            txt_handler.write('--- Configuration of PyTorch Training Routine ---\n')
            txt_handler.write(f'Date: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
            txt_handler.write(f'Used CPU: {self.used_hw_cpu}\n')
            txt_handler.write(f'Used GPU: {self.used_hw_gpu}\n')
            txt_handler.write(f'Used dataset: {self.data_settings.get_path2data()}\n')
            txt_handler.write(f'AI Topology: {self.settings.get_topology()} ({self._model_addon})\n')
            txt_handler.write(f'Embedded?: {self.model.model_embedded}\n')
            txt_handler.write('\n')
            txt_handler.write(f'Used Optimizer: {self.settings.optimizer}\n')
            txt_handler.write(f'Used Loss Function: {self.settings.loss}\n')
            txt_handler.write(f'Batchsize: {self.settings.batch_size}\n')
            txt_handler.write(f'Num. of epochs: {self.settings.num_epochs}\n')
            txt_handler.write(f'Splitting ratio (Training/Validation): '
                              f'{1-self.settings.data_split_ratio}/{self.settings.data_split_ratio}\n')
            txt_handler.write(f'Do KFold cross validation?: {self._do_kfold},\n'
                              f'Number of KFold steps: {self.settings.num_kfold}\n')
            txt_handler.write(f'Do shuffle?: {self.settings.data_do_shuffle}\n')
            txt_handler.write(f'Do data augmentation?: {self.data_settings.data_do_augmentation}\n')
            txt_handler.write(f'Do input normalization?: {self.data_settings.data_do_normalization}\n')
            txt_handler.write(f'Do add noise cluster?: {self.data_settings.data_do_addnoise_cluster}\n')
            txt_handler.write(f'Exclude cluster: {self.data_settings.data_exclude_cluster}\n')

    def _save_train_results(self, last_metric_train: float | np.ndarray,
                            last_metric_valid: float | np.ndarray, type='Loss') -> None:
        """Writing some training metrics into txt-file"""
        if self.config_available:
            with open(self._path2config, 'a') as txt_handler:
                txt_handler.write(f'\n--- Metrics of last epoch in fold #{self._run_kfold} ---')
                txt_handler.write(f'\nTraining {type} = {last_metric_train}')
                txt_handler.write(f'\nValidation {type} = {last_metric_valid}\n')

    def get_saving_path(self) -> str:
        """Getting the path for saving files in aim folder"""
        return self._path2save

    def get_best_model(self) -> list:
        """Getting the path to the best trained model"""
        return glob(join(self._path2save, "*.pth"))

    def _end_training_routine(self, timestamp_start: datetime, do_delete_temps=True) -> None:
        """Doing the last step of training routine"""
        timestamp_end = datetime.now()
        timestamp_string = timestamp_end.strftime('%H:%M:%S')
        diff_time = timestamp_end - timestamp_start
        diff_string = diff_time

        print(f'\nTraining ends on: {timestamp_string}')
        print(f'Training runs: {diff_string}')

        # Delete init model
        init_model = glob(join(self._path2save, 'model_reset.pth'))
        for file in init_model:
            remove(file)

        # Delete log folders
        if do_delete_temps:
            folder_logs = glob(join(self._path2save, 'temp*'))
            for folder in folder_logs:
                rmtree(folder, ignore_errors=True)

        # Give the option to open TensorBoard
        print("\nLook data on TensorBoard -> open Terminal")
        print("Type in: tensorboard serve --logdir ./runs")

    def get_data_points(self, num_output=4, use_train_dataloader=False) -> dict:
        """Getting data from DataLoader for Plotting Results"""
        output = [[] for _ in range(num_output)]
        keys = []
        mdict = dict()

        first_run = True
        for data_fold in (self.train_loader if use_train_dataloader else self.valid_loader):
            for vdata in data_fold:
                for idx, (key, value) in enumerate(vdata.items()):
                    output[idx] = value if first_run else np.append(output[idx], value, axis=0)
                    if key not in keys:
                        keys.append(key)
                first_run = False

        for key, value in zip(keys, output):
            mdict.update([(key, value)])

        return mdict
