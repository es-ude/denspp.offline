from dataclasses import dataclass
from typing import Any, SupportsInt
from os import getcwd, makedirs
from os.path import join, abspath, exists
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

    def print_overview_datasets(self, do_print=True) -> list:
        """"""
        oc_handler = ScieboDownloadHandler()
        list_datasets = oc_handler.get_overview_data()
        if do_print:
            print("\nNo local dataset is available. Enter the number of available datasets from remote:"
                  "\n==============================================================================")
            for idx, file in enumerate(list_datasets):
                print(f"\t{idx}: \t{file}")

        oc_handler.close()
        return list_datasets

    def load_dataset(self) -> dict:
        """Loading the dataset from defined data file"""
        self.__download_if_missing()
        return np.load(self.get_path2data, allow_pickle=True).flatten()[0]

    def __download_if_missing(self) -> None:
        """"""
        if self.data_file_name == '':
            list_datasets = self.print_overview_datasets(True)
            sel_data = input()
            self.data_file_name = list_datasets[int(sel_data)]
        else:
            list_datasets = self.print_overview_datasets(False)
            for file in list_datasets:
                if self.data_file_name.lower() in file.lower():
                    self.data_file_name = file
                    break

        if not exists(self.get_path2data):
            makedirs(self.get_path2folder, exist_ok=True)

            oc_handler = ScieboDownloadHandler()
            oc_handler.download_file(self.data_file_name, self.get_path2data)
            oc_handler.close()


class ScieboDownloadHandler:
    __oc_handler: owncloud.Client

    def __init__(self, link: str = 'https://uni-duisburg-essen.sciebo.de/s/JegLJuj1SADBSp0',
                 path2folder_remote: str = '/00_Merged/') -> None:
        """Class for handling sciebo repository for getting datasets remotely
        Args:
            link:                   String with link to used owncloud repository
            path2folder_remote:     Used folder on remote source
        Return:
            None
        """
        self.__public_sciebo_link = link
        self.__path2folder_remote = path2folder_remote

    def get_overview_data(self, formats: list = ('.npy', '.mat')) -> list:
        """Getting an overview of available files for downloading"""
        self.__oc_handler = owncloud.Client.from_public_link(self.__public_sciebo_link)
        dict_list = self.__oc_handler.list(self.__path2folder_remote, 1)
        self.__oc_handler.logout()

        files_available = list()
        for file in dict_list:
            for format in formats:
                if format in file.name:
                    files_available.append(file.name)
        return files_available

    def download_file(self, file_name: str, destination_download: str) -> None:
        """Downloading a file from remote server
        Args:
            file_name:  File name (for downloading remote file)
            destination_download:   Folder name to save the data locally
        Return:
            None
        """
        self.__oc_handler = owncloud.Client.from_public_link(self.__public_sciebo_link)
        print("... downloading file from sciebo")
        self.__oc_handler.get_file(join(self.__path2folder_remote, file_name), destination_download)
        print("... download done")

    def close(self) -> None:
        self.__oc_handler.logout()



DefaultSettingsDataset = Config_Dataset(
    data_path='data',
    data_file_name='quiroga',
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
    och = ScieboDownloadHandler()
    overview = och.get_overview_data()

    print(DefaultSettingsDataset.get_path2data)
    data = DefaultSettingsDataset.load_dataset()
    print(".done")
