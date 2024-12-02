from os import remove, getcwd, makedirs
from os.path import join
import platform
from copy import deepcopy
import cpuinfo
import numpy as np
from random import seed

from shutil import rmtree
from glob import glob
from datetime import datetime

from torch import (Tensor, is_tensor, zeros, unique, argwhere, device, cuda, backends, float32,
                   nn, randn, cat, Generator, manual_seed, use_deterministic_algorithms)
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from sklearn.model_selection import KFold

from package.dnn.pytorch_dataclass import Config_Dataset, Config_PyTorch
from package.structure_builder import create_folder_general_firstrun, create_folder_dnn_firstrun
from package.yaml_handler import translate_dataclass_to_dict, write_dict_to_yaml


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
        search_space = {'Classifier': '_cl', 'Autoencoder': '_ae', 'CNN+LSTM': '_2d'}

        addon = '_unknown'
        for key, addon in search_space.items():
            if self.out_modeltyp in key:
                break
        return addon

    @property
    def get_topology(self) -> str:
        """Getting the model name defined in models"""
        return self.out_modeltyp


class training_pytorch:
    deterministic_generator: Generator
    used_hw_dev: device
    used_hw_cpu: str
    used_hw_gpu: str
    used_hw_num: int
    train_loader: list
    valid_loader: list
    selected_samples: dict
    cell_classes: list
    _metric_methods: dict

    def __init__(self, config_train: Config_PyTorch, config_dataset: Config_Dataset,
                 do_train=True, do_print=True) -> None:
        """Class for Handling Training of Deep Neural Networks in PyTorch
        Args:
            config_train:   Configuration settings for the PyTorch Training
            config_dataset: Configuration settings for dataset handling
            do_train:       Mention if training should be used (default = True)
            do_print:       Printing the state and results into Terminal
        Returns:
            None
        """
        create_folder_general_firstrun()
        create_folder_dnn_firstrun()
        # --- Preparing Neural Network
        self.os_type = platform.system()
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        # --- Preparing options
        self.config_available = False
        self._kfold_do = False
        self._shuffle_do = config_train.data_do_shuffle
        self._kfold_run = 0
        # --- Saving options
        self.settings_train = config_train
        self.settings_data = config_dataset
        self._index_folder = 'train' if do_train else 'inference'
        self._do_print_state = do_print
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
        makedirs(path2dst, exist_ok=True)


    def __setup_device(self) -> None:
        """Setup PyTorch for Training"""
        self.used_hw_cpu = (f"{cpuinfo.get_cpu_info()['brand_raw']} "
                            f"(@ {1e-9 * cpuinfo.get_cpu_info()['hz_actual'][0]:.3f} GHz)")

        if cuda.is_available():
            # Using GPU
            self.used_hw_gpu = cuda.get_device_name()
            self.used_hw_dev = device("cuda")
            self.used_hw_num = cuda.device_count()
            device0 = self.used_hw_gpu
            cuda.empty_cache()
        elif backends.mps.is_available() and backends.mps.is_built() and self.os_type == "Darwin":
            # Using Apple M1 Chip
            self.used_hw_gpu = 'None'
            self.used_hw_num = cuda.device_count()
            self.used_hw_dev = device("mps")
            device0 = self.used_hw_cpu
        else:
            # Using normal CPU
            self.used_hw_gpu = 'None'
            self.used_hw_dev = device("cpu")
            self.used_hw_num = 1 # cpuinfo.get_cpu_info()['count']
            device0 = self.used_hw_cpu

        if self._do_print_state:
            print(f"\nUsing PyTorch with {device0} on {self.os_type}")

    def _init_train(self, path2save='', addon='') -> None:
        """Do init of class for training"""
        # --- Generate links
        if not path2save:
            folder_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self._index_folder}_{self._model_name}'
            self._path2save = join(self._path2run, folder_name)
        else:
            self._path2save = path2save
        self._path2temp = join(self._path2save, f'temp')

        # --- Generate folders
        makedirs(self._path2run, exist_ok=True)
        makedirs(self._path2save, exist_ok=True)
        makedirs(self._path2temp, exist_ok=True)

        # --- Transfer model to hardware
        self.model.to(device=self.used_hw_dev)

        # --- Copy settings to YAML file
        write_dict_to_yaml(translate_dataclass_to_dict(self.settings_data),
                           filename='Config_Dataset', path2save=self._path2save)
        write_dict_to_yaml(translate_dataclass_to_dict(self.settings_train),
                           filename=f'Config_Training{addon}', path2save=self._path2save)

    def __deterministic_training_preparation(self) -> None:
        """Preparing the CUDA hardware for deterministic training"""
        if self.settings_train.deterministic_do:
            np.random.seed(self.settings_train.deterministic_seed)
            manual_seed(self.settings_train.deterministic_seed)
            if cuda.is_available():
                cuda.manual_seed_all(self.settings_train.deterministic_seed)
            seed(self.settings_train.deterministic_seed)
            backends.cudnn.deterministic = True

            use_deterministic_algorithms(True)
            if self._do_print_state:
                print(f"\n\t\t=== DL Training with Deterministic @seed: {self.settings_train.deterministic_seed} ===")
        else:
            use_deterministic_algorithms(False)
            if self._do_print_state:
                print(f"\n\t\t=== Normal DL Training ===")

    def __deterministic_get_dataloader_params(self) -> dict:
        """Getting the parameters for preparing the Training and Validation DataLoader for Deterministic Training"""
        if self.settings_train.deterministic_do:
            self.deterministic_generator = Generator()
            self.deterministic_generator.manual_seed(self.settings_train.deterministic_seed)
            worker_init_fn = lambda worker_id: np.random.seed(self.settings_train.deterministic_seed)
            return {'worker_init_fn': worker_init_fn, 'generator': self.deterministic_generator}
        else:
            return {}

    def load_data(self, data_set, num_workers=0) -> None:
        """Loading data for training and validation in DataLoader format into class
        Args:
            data_set:       DataLoader of used dataset
            num_workers:    Number of workers for calculation [Default: 0 --> single core]
        Return:
            None
        """
        self.__setup_device()
        self._kfold_do = True if self.settings_train.num_kfold > 1 else False
        self._model_addon = data_set.get_topology_type
        self.cell_classes = data_set.get_dictionary
        params_deterministic = self.__deterministic_get_dataloader_params()

        # --- Preparing datasets
        out_train = list()
        out_valid = list()
        # ToDo: SubsetRandomSampler works valid with deterministic training? - Please check!
        if self._kfold_do:
            kfold = KFold(n_splits=self.settings_train.num_kfold,
                          shuffle=self._shuffle_do and not self.settings_train.deterministic_do)
            for idx_train, idx_valid in kfold.split(np.arange(len(data_set))):
                subsamps_train = SubsetRandomSampler(idx_train)
                subsamps_valid = SubsetRandomSampler(idx_valid)
                out_train.append(DataLoader(data_set,
                                            batch_size=self.settings_train.batch_size,
                                            sampler=subsamps_train,
                                            **params_deterministic))
                out_valid.append(DataLoader(data_set,
                                            batch_size=self.settings_train.batch_size,
                                            sampler=subsamps_valid,
                                            **params_deterministic))
        else:
            idx = np.arange(len(data_set))
            if self._shuffle_do and not self.settings_train.deterministic_do:
                np.random.shuffle(idx)
            split_pos = int(len(data_set) * (1 - self.settings_train.data_split_ratio))
            idx_train = idx[0:split_pos]
            idx_valid = idx[split_pos:]
            subsamps_train = SubsetRandomSampler(idx_train)
            subsamps_valid = SubsetRandomSampler(idx_valid)
            out_train.append(DataLoader(data_set,
                                        batch_size=self.settings_train.batch_size,
                                        sampler=subsamps_train,
                                        **params_deterministic))
            out_valid.append(DataLoader(data_set,
                                        batch_size=self.settings_train.batch_size,
                                        sampler=subsamps_valid,
                                        **params_deterministic))

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

        # --- Init. hardware for deterministic training
        if self.settings_train.deterministic_do:
            self.__deterministic_training_preparation()

        # --- Print model
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
            txt_handler.write(f'AI Topology: {self.model.get_topology}\n')
            txt_handler.write(f'Used Dataset: {self._model_addon}\n')
            txt_handler.write(f'Embedded?: {self.model.model_embedded}\n')
            txt_handler.write('\n')
            txt_handler.write(f'Used Optimizer: {self.settings_train.optimizer}\n')
            txt_handler.write(f'Used Loss Function: {self.settings_train.loss}\n')
            txt_handler.write(f'Batchsize: {self.settings_train.batch_size}\n')
            txt_handler.write(f'Num. of epochs: {self.settings_train.num_epochs}\n')
            txt_handler.write(f'Splitting ratio (Training/Validation): '
                              f'{1-self.settings_train.data_split_ratio}/{self.settings_train.data_split_ratio}\n')
            txt_handler.write(f'Do KFold cross validation?: {self._kfold_do},\n'
                              f'Number of KFold steps: {self.settings_train.num_kfold}\n')
            txt_handler.write(f'Do shuffle?: {self.settings_train.data_do_shuffle}\n')
            txt_handler.write(f'Do data augmentation?: {self.settings_data.augmentation_do}\n')
            txt_handler.write(f'Do input normalization?: {self.settings_data.normalization_do}\n')
            txt_handler.write(f'Exclude cluster: {self.settings_data.exclude_cluster}\n')
            txt_handler.write(f'Do deterministic training: {self.settings_train.deterministic_do}\n')
            txt_handler.write(f'Deterministic Training, Seed: {self.settings_train.deterministic_seed}\n')

    def _save_train_results(self, last_metric_train: float | np.ndarray,
                            last_metric_valid: float | np.ndarray, loss_type='Loss') -> None:
        """Writing some training metrics into txt-file"""
        if self.config_available:
            with open(self._path2config, 'a') as txt_handler:
                txt_handler.write(f'\n--- Metrics of last epoch in fold #{self._kfold_run} ---')
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

        if self._do_print_state:
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

    def _getting_data_for_plotting(self, valid_input: np.ndarray, valid_label: np.ndarray,
                                   results=None, addon='cl') -> dict:
        """Getting the raw data for plotting results"""
        # --- Producing and Saving the output
        if results is None:
            results = dict()

        if self._do_print_state:
            print(f"... preparing results for plot generation")
        data_train = self.__get_data_points(only_getting_labels=True, use_train_dataloader=True)

        output = dict()
        output.update({'settings': self.settings_train, 'date': datetime.now().strftime('%d/%m/%Y, %H:%M:%S')})
        output.update({'train_clus': data_train['class'] if addon == 'ae' else data_train['out'], 'cl_dict': self.cell_classes})
        output.update({'input': valid_input, 'valid_clus': valid_label})
        output.update(results)

        data2save = join(self.get_saving_path(), f'results_{addon}.npy')
        if self._do_print_state:
            print(f"... saving results: {data2save}")
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

    def _separate_classes_from_label(self, pred: Tensor, true: Tensor, label: str, *args) -> [Tensor, Tensor]:
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
            metric_out = zeros((len(self.cell_classes),), dtype=float32)
        else:
            metric_out = [zeros((1,)) for _ in self.cell_classes]

        length_out = zeros((len(self.cell_classes),), dtype=float32)
        for idx, id in enumerate(unique(true)):
            xpos = argwhere(true == id).flatten()
            length_out[idx] = len(xpos)
            if args:
                metric_out[idx] += args[0](pred[xpos], true[xpos])
            else:
                metric_out[idx] = pred[xpos]
        return metric_out, length_out

    def _converting_tensor_to_numpy(self, metric_used: dict) -> dict:
        """Converting tensor array to numpy for later processing"""
        # --- Metric out for saving (converting from tensor to numpy)
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

    def get_metric_methods(self) -> None:
        """Function for calling the functions to calculate metrics during training phase"""
        print(self._metric_methods.keys())

    @property
    def get_number_parameters_from_model(self) -> int:
        """Getting the number of used parameters of used DNN model"""
        return int(sum(p.numel() for p in self.model.parameters()))
