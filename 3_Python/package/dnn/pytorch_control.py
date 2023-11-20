import dataclasses
from typing import Any
from os import mkdir
from os.path import exists, join
from glob import glob
import platform
import numpy as np
from datetime import datetime
from torch import optim, device, cuda
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from sklearn.model_selection import KFold


@dataclasses.dataclass(frozen=True)
class Config_PyTorch:
    """Template for configurating pytorch for training a model"""
    # --- Settings of Models/Training
    model: Any
    loss_fn: Any
    optimizer: str
    num_kfold: int
    num_epochs: int
    batch_size: int
    # --- Settings of Datasets
    data_path: str
    data_file_name: str
    data_split_ratio: float
    data_do_shuffle: bool
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

    def get_topology(self) -> str:
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


class training_pytorch:
    """Class for Handling Training of Deep Neural Networks in PyTorch"""
    def __init__(self, config_train: Config_PyTorch, do_train=True) -> None:
        self.device = None
        self.os_type = None
        self._writer = None
        self.__setup_device()

        # --- Preparing options
        self._do_kfold = False
        self._run_kfold = 0

        # --- Saving options
        self._index_folder = 'train' if do_train else 'inference'
        self._aitype = config_train.model.out_modeltyp
        self._model_name = config_train.model.out_modelname
        self._model_addon = str()
        self._path2run = 'runs'
        self._path2log = str()
        self._path2config = str()
        self.config_available = False
        self._path2save = str()

        # --- Training input
        self.settings = config_train
        self.model = self.settings.model
        self._used_model = None
        self.loss_fn = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None

    def __setup_device(self) -> None:
        """Setup PyTorch for Training"""
        device0 = "CUDA" if cuda.is_available() else "CPU"
        if device0 == "CUDA":
            self.device = device("cuda")
        else:
            self.device = device("cpu")

        self.os_type = platform.system()

        print(f"... using PyTorch with {device0} device on {self.os_type}")

    def _init_train(self) -> None:
        """Do init of class for training"""
        folder_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self._index_folder}_{self._model_name}'
        self._path2save = join(self._path2run, folder_name)

        if not exists(self._path2run):
            mkdir(self._path2run)

        mkdir(self._path2save)

    def _init_writer(self) -> None:
        """Do init of writer"""
        self._path2log = join(self._path2save, f'logs_{self._run_kfold:03d}')
        self._writer = SummaryWriter(self._path2log)

    def load_data(self, data_set) -> None:
        """Loading data for training and validation in DataLoader format into class"""
        self._do_kfold = True if self.settings.num_kfold > 1 else False
        num_samples = len(data_set)
        self._model_addon = data_set.data_type

        # --- Preparing datasets
        out_train = list()
        out_valid = list()
        if self._do_kfold:
            kfold = KFold(n_splits=self.settings.num_kfold, shuffle=True)
            for fold, (idx_train, idx_valid) in enumerate(kfold.split(np.arange(num_samples))):
                subsamps_train = SubsetRandomSampler(idx_train)
                subsamps_valid = SubsetRandomSampler(idx_valid)
                out_train.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_train))
                out_valid.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_valid))
        else:
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            pos = int(num_samples * (1 - self.settings.data_split_ratio))
            idx_train = idx[0:pos]
            idx_valid = idx[pos:]
            subsamps_train = SubsetRandomSampler(idx_train)
            subsamps_valid = SubsetRandomSampler(idx_valid)

            out_train.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_train))
            out_valid.append(DataLoader(data_set, batch_size=self.settings.batch_size, sampler=subsamps_valid))

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

    def _save_config_txt(self) -> None:
        """Writing the content of the configuration class in *.txt-file"""
        self._path2config = join(self._path2save, 'config.txt')
        self.config_available = True

        with open(self._path2config, 'w') as txt_handler:
            txt_handler.write('--- Configuration of PyTorch Training Routine ---\n')
            txt_handler.write(f'Date: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
            txt_handler.write(f'AI Topology: {self.settings.get_topology()} ({self._model_addon})\n')
            txt_handler.write(f'Embedded?: {self.model.model_embedded}\n')
            txt_handler.write('\n')
            txt_handler.write(f'Used Optimizer: {self.settings.optimizer}\n')
            txt_handler.write(f'Batchsize: {self.settings.batch_size}\n')
            txt_handler.write(f'Num. of epochs: {self.settings.num_epochs}\n')
            txt_handler.write(f'Splitting ratio (Training/Validation): '
                              f'{1-self.settings.data_split_ratio}/{self.settings.data_split_ratio}\n')
            txt_handler.write(f'Do kfold cross validation?: {self._do_kfold}, '
                              f'Number of steps: {self.settings.num_kfold}\n')
            txt_handler.write(f'Do shuffle?: {self.settings.data_do_shuffle}\n')
            txt_handler.write(f'Do data augmentation?: {self.settings.data_do_augmentation}\n')
            txt_handler.write(f'Do input normalization?: {self.settings.data_do_normalization}\n')
            txt_handler.write(f'Do add noise cluster?: {self.settings.data_do_addnoise_cluster}\n')
            txt_handler.write(f'Exclude cluster: {self.settings.data_exclude_cluster}\n')

    def _save_train_results(self, last_metric_train: float, last_metric_valid: float, type='Loss') -> None:
        """Writing some training metrics into txt-file"""
        if self.config_available:
            with open(self._path2config, 'a') as txt_handler:
                txt_handler.write('\n--- Results of last epoch ---')
                txt_handler.write(f'\nTraining {type} = {last_metric_train}')
                txt_handler.write(f'\nValidation {type} = {last_metric_valid}')

    def get_saving_path(self) -> str:
        """Getting the path for saving files in aim folder"""
        return self._path2save

    def get_best_model(self) -> list:
        """Getting the path to the best trained model"""
        return glob(join(self._path2save, "*.pth"))

