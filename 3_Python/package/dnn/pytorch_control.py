from os import mkdir
from os.path import exists, join
import platform
import numpy as np
from datetime import datetime
from torch import nn, optim, device, cuda
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from sklearn.model_selection import KFold


class Config_PyTorch:
    """Template for configurating pytorch for training a model"""
    def __init__(self):
        # Settings of Models/Training
        self.model = "model name"   # example: ai_module.cnn_ae_v1
        self.is_embedded = False
        self.loss_fn = nn.MSELoss()
        self.num_kfold = 1
        self.num_epochs = 1000
        self.batch_size = 512
        # Settings of Datasets
        self.data_path = 'data'
        self.data_file_name = '2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted'
        self.data_split_ratio = 0.2
        self.data_do_shuffle = True
        self.data_do_augmentation = True
        self.data_num_augmentation = 2000
        self.data_do_normalization = False
        self.data_do_addnoise_cluster = False
        self.data_do_reduce_samples_per_cluster = True
        self.data_num_samples_per_cluster = 10000
        # Dataset Preparation
        self.data_exclude_cluster = [1]
        self.data_sel_pos = []

    def set_optimizer(self, model):
        return optim.Adam(model.parameters())

    def get_topology(self, model) -> str:
        return model.out_modeltyp


class training_pytorch:
    """Class for Handling Training of Deep Neural Networks in PyTorch"""
    def __init__(self, type: str, model_name: str, config_train: Config_PyTorch, do_train=True) -> None:
        self.device = None
        self.os_type = None
        self._writer = None
        self.__setup_device()

        # --- Preparing options
        self._do_kfold = False
        self._run_kfold = 0

        # --- Saving options
        self._index_folder = 'train' if do_train else 'inference'
        self._aitype = type
        self._model_name = model_name
        self._model_addon = str()
        self._path2run = 'runs'
        self._path2log = str()
        self._path2config = str()
        self.config_available = False
        self._path2save = str()

        # --- Training input
        self.settings = config_train
        self.model = None
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
        self.__preparing_data(data_set)

    def __preparing_data(self, data_set) -> None:
        """Preparing data for training"""
        self._do_kfold = True if self.settings.num_kfold > 1 else False
        num_samples = len(data_set)

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

    def load_model(self, model: nn.Module, optimizer, print_model=True) -> None:
        """Loading model, optimizer, loss_fn into class"""
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = self.settings.loss_fn
        if print_model:
            summary(self.model, input_size=self.model.model_shape)

    def _save_config_txt(self) -> None:
        """Writing the content of the configuration class in *.txt-file"""
        config_handler = self.settings
        self._path2config = join(self._path2save, 'config.txt')
        self.config_available = True

        with open(self._path2config, 'w') as txt_handler:
            txt_handler.write('--- Configuration of PyTorch Training Routine ---\n')
            txt_handler.write(f'Date: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
            txt_handler.write(f'AI Topology: {config_handler.get_topology(self.model) + self._model_addon}\n')
            txt_handler.write(f'Embedded?: {self.model.model_embedded}\n')
            txt_handler.write('\n')
            txt_handler.write(f'Batchsize: {config_handler.batch_size}\n')
            txt_handler.write(f'Num. of epochs: {config_handler.num_epochs}\n')
            txt_handler.write(f'Splitting ratio (Training/Validation): '
                              f'{1-config_handler.data_split_ratio}/{config_handler.data_split_ratio}\n')
            txt_handler.write(f'Do kfold cross validation?: {self._do_kfold}, '
                              f'Number of steps: {self.settings.num_kfold}\n')
            txt_handler.write(f'Do shuffle?: {config_handler.data_do_shuffle}\n')
            txt_handler.write(f'Do data augmentation?: {config_handler.data_do_augmentation}\n')
            txt_handler.write(f'Do input normalization?: {config_handler.data_do_normalization}\n')
            txt_handler.write(f'Do add noise cluster?: {config_handler.data_do_addnoise_cluster}\n')
            txt_handler.write(f'Exclude cluster: {config_handler.data_exclude_cluster}\n')

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
