from dataclasses import dataclass
from typing import Any
from os import getcwd
from os.path import join
from torch import optim, nn


@dataclass
class Config_PyTorch:
    """Class for handling the PyTorch training/inference pipeline"""
    model_name: str
    patience: int
    optimizer: str
    loss: str
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
    data_split_ratio=0.2
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
    normalization_mode: str
    normalization_method: str
    normalization_setting: str
    reduce_samples_per_cluster_do: bool
    reduce_samples_per_cluster_num: int
    # --- Dataset Preparation
    exclude_cluster: list

    @property
    def get_path2data(self) -> str:
        """Getting the path name to the file"""
        return join(self.data_path if not self.data_path == '' else self.get_path2data_default, self.data_file_name)

    @property
    def get_path2data_default(self, start_folder='3_Python') -> str:
        """Getting the default path to data from repository"""
        return join(getcwd().split(start_folder)[0], '2_Data', '00_Merged_Datasets')


DefaultSettingsDataset = Config_Dataset(
    data_path='',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    use_cell_library=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=False,
    normalization_mode='CPU',
    normalization_method='minmax',
    normalization_setting='bipolar',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[]
)


if __name__ == "__main__":
    print(DefaultSettingsDataset.get_path2data_default)
    print(DefaultSettingsDataset.get_path2data)
