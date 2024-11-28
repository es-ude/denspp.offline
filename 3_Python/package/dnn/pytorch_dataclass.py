from dataclasses import dataclass
from typing import Any
from os import getcwd, makedirs
from os.path import join, abspath, exists
from pathlib import Path
from torch import optim, nn
import numpy as np
import owncloud


@dataclass
class Config_PyTorch:
    """Class for handling the PyTorch training/inference pipeline"""
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

    def get_loss_func(self) -> Any:
        """Getting the loss function"""
        match self.loss:
            case 'MSE':
                loss_func = nn.MSELoss()
            case 'Cross Entropy':
                loss_func = nn.CrossEntropyLoss()
            case _:
                raise NotImplementedError("Loss function unknown! - Please implement or check!")
        return loss_func

    def load_optimizer(self, model, learn_rate=0.1) -> Any:
        """Loading the optimizer function"""
        match self.optimizer:
            case 'Adam':
                optim_func = self.__set_optimizer_adam(model)
            case 'SGD':
                optim_func = self.__set_optimizer_sgd(model, learn_rate=learn_rate)
            case _:
                raise NotImplementedError("Optimizer function unknown! - Please implement or check!")
        return optim_func

    def __set_optimizer_adam(self, model):
        """Using the Adam Optimizer"""
        return optim.Adam(model.parameters())

    def __set_optimizer_sgd(self, model, learn_rate=0.1):
        """Using the SGD as Optimizer"""
        return optim.SGD(model.parameters(), lr=learn_rate)


DefaultSettingsTrainMSE = Config_PyTorch(
    model_name='dnn_ae_v1',
    patience=20,
    optimizer='Adam',
    loss='MSE',
    deterministic_do=False,
    deterministic_seed=42,
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.2
)
DefaultSettingsTrainCE = Config_PyTorch(
    model_name='dnn_cl_v1',
    patience=20,
    optimizer='Adam',
    loss='Cross Entropy',
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.2,
    deterministic_do=False,
    deterministic_seed=42
)


@dataclass
class Config_Dataset:
    """Class for handling preparation of dataset"""
    # --- Settings of Datasets
    data_path: str
    data_file_name: str
    use_cell_library: int
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
        return abspath(join(self.get_path2folder, self.data_file_name))

    @property
    def get_path2folder(self) -> str:
        """Getting the path name to the file"""
        if self.data_path == 'data':
            path = join(self.get_path2folder_data)
        elif not self.data_path == '':
            path = join(self.get_path2folder_project, self.data_path)
        else:
            path = join(self.data_path)
        return abspath(path)

    @property
    def get_path2folder_data(self) -> str:
        """Getting the default path of the data inside the Python Project"""
        return abspath(join(self.get_path2folder_project, 'data'))

    @property
    def get_path2folder_project(self, start_folder='3_Python') -> str:
        """Getting the default path of the Python Project"""
        return abspath(join(getcwd().split(start_folder)[0], start_folder))

    @property
    def get_path2folder_examples(self, start_folder='3_Python') -> str:
        """Getting the default path to data from repository"""
        return abspath(join(getcwd().split(start_folder)[0], '2_Data', '00_Merged_Datasets'))

    def load_dataset(self) -> dict:
        """Loading the dataset from defined data file"""
        valid_filenames = ['martinez', 'quiroga', 'sda', 'rgc_tdb', 'fzj_mcs']
        self.__download_if_missing()

        valid_file_exists = False
        for key in valid_filenames:
            if key in self.get_path2data.lower():
                valid_file_exists = True
                break
        if valid_file_exists:
            return np.load(self.get_path2data, allow_pickle=True).flatten()[0]
        else:
            raise NotImplementedError("Dataset Structure is not defined - Please check!")

    def __download_if_missing(self) -> None:
        """"""
        if not exists(self.get_path2data):
            public_link = 'https://uni-duisburg-essen.sciebo.de/s/JegLJuj1SADBSp0'
            path2folder_remote = '/00_Merged/'

            makedirs(self.get_path2folder, exist_ok=True)
            path_remote_file = path2folder_remote + self.data_file_name

            oc = owncloud.Client.from_public_link(public_link)
            print("... downloading file from sciebo")
            oc.get_file(path_remote_file, self.get_path2data)
            print("... download done")
            oc.logout()


DefaultSettingsDataset = Config_Dataset(
    data_path='',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.npy',
    use_cell_library=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=False,
    normalization_method='bipolar',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[]
)


if __name__ == "__main__":
    print(DefaultSettingsDataset.get_path2data)
