from os import mkdir, remove, getcwd
from os.path import exists, join
import platform
import cpuinfo
import numpy as np

from shutil import rmtree
from glob import glob
from datetime import datetime
from torch import device, cuda, backends, nn, randn, cat
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from sklearn.model_selection import KFold

from package.dnn.pytorch_dataclass import Config_PyTorch, Config_Dataset
from package.structure_builder import create_folder_general_firstrun, create_folder_dnn_firstrun


class ModelRegistry:
    def __init__(self):
        """Class for building the overview of neural networks"""
        self.data = {}

    def register(self, fn):
        """Adding a class with neural network topology to system"""
        self.data[fn.__name__] = fn
        return fn

    def build_model(self, name: str, *args, **kwargs):
        """Build the model"""
        model = self.data[name](*args, **kwargs)
        print(model.__annotations__)
        return model

    def get_model_overview(self, do_print=True) -> list:
        """Getting an overview of existing models in library"""
        if do_print:
            print("\nOverview of available neural network models"
                  "\n====================================================")
            idx = 0
            for key, func in self.data.items():
                print(f"\t#{idx:02d}: {key}")
                # print(func.__annotations__)
                idx += 1

        return [key for key in self.data.keys()]


class __model_settings_common(nn.Module):
    model: nn.Sequential

    def __init__(self, type_model: str):
        """"""
        super().__init__()
        self.model_shape = (1, 28, 28)
        self.model_embedded = False
        self.out_modeltyp = type_model
        self.out_modelname = self.get_modelname

    @property
    def get_modelname(self) -> str:
        """Getting the name of the model"""
        return self.__class__.__name__ + self.__get_addon

    @property
    def __get_addon(self) -> str:
        """Getting the prefix / addon of used network topology"""
        match self.out_modeltyp:
            case 'Classifier':
                addon = '_cl'
            case 'Autoencoder':
                addon = '_ae'
            case _:
                addon = '_unknown'
        return addon

    @property
    def get_topology(self) -> str:
        """Getting the model name defined in models"""
        return self.out_modeltyp


class training_pytorch:
    used_hw_dev: device
    used_hw_cpu: str
    used_hw_gpu: str
    used_hw_num: int
    train_loader: list
    valid_loader: list
    selected_samples: dict
    cell_classes: list

    def __init__(self, config_train: Config_PyTorch, config_dataset: Config_Dataset, do_train=True) -> None:
        """Class for Handling Training of Deep Neural Networks in PyTorch
        Args:
            config_train:   Configuration settings for the PyTorch Training
            config_dataset: Configuration settings for dataset handling
            do_train:       Mention if training should be used (default = True)
        Returns:
            None
        """
        create_folder_general_firstrun()
        create_folder_dnn_firstrun()
        # --- Preparing Neural Network
        self.os_type = platform.system()
        self._writer = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        # --- Preparing options
        self.config_available = False
        self._do_kfold = False
        self._do_shuffle = config_train.data_do_shuffle
        self._run_kfold = 0
        # --- Saving options
        self.settings_train = config_train
        self.settings_data = config_dataset
        self._index_folder = 'train' if do_train else 'inference'
        self._aitype = str()
        self._model_name = str()
        self._model_addon = str()
        # --- Logging paths for saving
        self.__check_start_folder()
        self._path2save = str()
        self._path2log = str()
        self._path2temp = str()
        self._path2config = str()

    def __check_start_folder(self, start_folder='3_Python', new_folder='runs'):
        """Checking for starting folder to generate"""
        path2start = join(getcwd().split(start_folder)[0], start_folder)
        path2dst = join(path2start, new_folder)
        self._path2run = path2dst
        if not exists(path2dst):
            mkdir(path2dst)

    def __setup_device(self) -> None:
        """Setup PyTorch for Training"""
        self.used_hw_cpu = (f"{cpuinfo.get_cpu_info()['brand_raw']} "
                            f"(@ {1e-9 * cpuinfo.get_cpu_info()['hz_actual'][0]:.3f} GHz)")
        # Using GPU
        if cuda.is_available():
            self.used_hw_gpu = cuda.get_device_name()
            self.used_hw_dev = device("cuda")
            self.used_hw_num = cuda.device_count()
            device0 = self.used_hw_gpu
            cuda.empty_cache()
        # Using Apple M1 Chip
        elif backends.mps.is_available() and backends.mps.is_built() and self.os_type == "Darwin":
            self.used_hw_gpu = 'None'
            self.used_hw_num = cuda.device_count()
            self.used_hw_dev = device("mps")
            self.used_hw_num = 1
            device0 = self.used_hw_cpu
        # Using normal CPU
        else:
            self.used_hw_gpu = 'None'
            self.used_hw_dev = device("cpu")
            self.used_hw_num = cpuinfo.get_cpu_info()['count']
            device0 = self.used_hw_cpu
        print(f"\nUsing PyTorch with {device0} on {self.os_type}")

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

        self.model.to(device=self.used_hw_dev)

    def _init_writer(self) -> None:
        """Do init of writer"""
        self._path2log = join(self._path2save, f'logs')
        self._writer = SummaryWriter(self._path2log, comment=f"event_log_kfold{self._run_kfold:03d}")

    def load_data(self, data_set, num_workers=0) -> None:
        """Loading data for training and validation in DataLoader format into class"""
        self.__setup_device()
        self._do_kfold = True if self.settings_train.num_kfold > 1 else False
        self._model_addon = data_set.data_type
        self.cell_classes = data_set.frame_dict if data_set.cluster_name_available else []

        # --- Preparing datasets
        out_train = list()
        out_valid = list()
        if self._do_kfold:
            kfold = KFold(n_splits=self.settings_train.num_kfold, shuffle=self._do_shuffle)
            for idx_train, idx_valid in kfold.split(np.arange(len(data_set))):
                subsamps_train = SubsetRandomSampler(idx_train)
                subsamps_valid = SubsetRandomSampler(idx_valid)
                out_train.append(DataLoader(data_set, batch_size=self.settings_train.batch_size, sampler=subsamps_train))
                out_valid.append(DataLoader(data_set, batch_size=self.settings_train.batch_size, sampler=subsamps_valid))
        else:
            idx = np.arange(len(data_set))
            if self._do_shuffle:
                np.random.shuffle(idx)
            split_pos = int(len(data_set) * (1 - self.settings_train.data_split_ratio))
            idx_train = idx[0:split_pos]
            idx_valid = idx[split_pos:]
            subsamps_train = SubsetRandomSampler(idx_train)
            subsamps_valid = SubsetRandomSampler(idx_valid)
            out_train.append(DataLoader(data_set, batch_size=self.settings_train.batch_size, sampler=subsamps_train))
            out_valid.append(DataLoader(data_set, batch_size=self.settings_train.batch_size, sampler=subsamps_valid))

        # --- CUDA support for dataset
        if cuda.is_available():
            for idx, dataset in enumerate(out_train):
                out_train[idx].pin_memory = True
                out_train[idx].pin_memory_device = self.used_hw_dev.type
                out_train[idx].num_workers = num_workers

                out_valid[idx].pin_memory = True
                out_valid[idx].pin_memory_device = self.used_hw_dev.type
                out_valid[idx].num_workers = num_workers

        # --- Output: Data
        self.train_loader = out_train
        self.valid_loader = out_valid

    def load_model(self, model, learn_rate=0.1, print_model=True) -> None:
        """Loading optimizer, loss_fn into class
        Args:
            model:          PyTorch Neural Network for Training / Inference
            learn_rate:     Learning rate used for SGD optimier
            print_model:    Print the model summary [Default: True]
        Returns:
            None
        """
        self.model = model
        self._aitype = model.out_modeltyp
        self._model_name = model.out_modelname

        self.optimizer = self.settings_train.load_optimizer(model, learn_rate=learn_rate)
        self.loss_fn = self.settings_train.get_loss_func()
        if print_model:
            print("\nPrint summary of model")
            summary(self.model, input_size=self.model.model_shape)
            print("\n\n")

    def _save_config_txt(self, addon='') -> None:
        """Writing the content of the configuration class in *.txt-file"""
        self._path2config = join(self._path2save, f'config{addon}.txt')
        self.config_available = True

        with open(self._path2config, 'w') as txt_handler:
            txt_handler.write('--- Configuration of PyTorch Training Routine ---\n')
            txt_handler.write(f'Date: {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
            txt_handler.write(f'Used CPU: {self.used_hw_cpu}\n')
            txt_handler.write(f'Used GPU: {self.used_hw_gpu}\n')
            txt_handler.write(f'Used dataset: {self.settings_data.get_path2data}\n')
            txt_handler.write(f'AI Topology: {self.model.get_topology} ({self._model_addon})\n')
            txt_handler.write(f'Embedded?: {self.model.model_embedded}\n')
            txt_handler.write('\n')
            txt_handler.write(f'Used Optimizer: {self.settings_train.optimizer}\n')
            txt_handler.write(f'Used Loss Function: {self.settings_train.loss}\n')
            txt_handler.write(f'Batchsize: {self.settings_train.batch_size}\n')
            txt_handler.write(f'Num. of epochs: {self.settings_train.num_epochs}\n')
            txt_handler.write(f'Splitting ratio (Training/Validation): '
                              f'{1-self.settings_train.data_split_ratio}/{self.settings_train.data_split_ratio}\n')
            txt_handler.write(f'Do KFold cross validation?: {self._do_kfold},\n'
                              f'Number of KFold steps: {self.settings_train.num_kfold}\n')
            txt_handler.write(f'Do shuffle?: {self.settings_train.data_do_shuffle}\n')
            txt_handler.write(f'Do data augmentation?: {self.settings_data.augmentation_do}\n')
            txt_handler.write(f'Do input normalization?: {self.settings_data.normalization_do}\n')
            txt_handler.write(f'Do add noise cluster?: {self.settings_data.add_noise_cluster}\n')
            txt_handler.write(f'Exclude cluster: {self.settings_data.exclude_cluster}\n')

    def _save_train_results(self, last_metric_train: float | np.ndarray,
                            last_metric_valid: float | np.ndarray, loss_type='Loss') -> None:
        """Writing some training metrics into txt-file"""
        if self.config_available:
            with open(self._path2config, 'a') as txt_handler:
                txt_handler.write(f'\n--- Metrics of last epoch in fold #{self._run_kfold} ---')
                txt_handler.write(f'\nTraining {loss_type} = {last_metric_train}')
                txt_handler.write(f'\nValidation {loss_type} = {last_metric_valid}\n')

    def get_saving_path(self) -> str:
        """Getting the path for saving files in aim folder"""
        return self._path2save

    def get_best_model(self, type_model: str) -> list:
        """Getting the path to the best trained model"""
        return glob(join(self._path2save, f"*{type_model}*.pth"))

    def _end_training_routine(self, timestamp_start: datetime, do_delete_temps=True) -> None:
        """Doing the last step of training routine"""
        timestamp_end = datetime.now()
        timestamp_string = timestamp_end.strftime('%H:%M:%S')
        diff_time = timestamp_end - timestamp_start
        diff_string = diff_time

        print(f'\nTraining ends on: {timestamp_string}')
        print(f'Training runs: {diff_string}')

        # Delete init model
        init_model = glob(join(self._path2save, '*_reset.pth'))
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

    def __get_data_points(self, only_getting_labels=False, use_train_dataloader=False) -> dict:
        """Getting data from DataLoader for Plotting Results
        Args:
            only_getting_labels:    Option for taking only labels
            use_train_dataloader:   Mode for selecting datatype (True=Training, False=Validation)
        Returns:
              Dict with data for plotting
        """
        used_dataset = self.train_loader[-1] if use_train_dataloader else self.valid_loader[-1]

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

    def _getting_data_for_plotting(self, valid_input: np.ndarray, valid_label: np.ndarray, results=None) -> dict:
        """Getting the raw data for plotting results"""
        # --- Producing and Saving the output
        if results is None:
            results = {}

        print(f"... preparing results for plot generation")
        data_train = self.__get_data_points(only_getting_labels=True, use_train_dataloader=True)

        output = dict()
        output.update({'settings': self.settings_train, 'date': datetime.now().strftime('%d/%m/%Y, %H:%M:%S')})
        output.update({'train_clus': data_train['out'], 'cl_dict': self.cell_classes})
        output.update({'input': valid_input, 'valid_clus': valid_label})
        output.update(results)

        data2save = join(self.get_saving_path(), 'results_class.npy')
        print(f"... saving results: {data2save}")
        np.save(data2save, output)
        return output
