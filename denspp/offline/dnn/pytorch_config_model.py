from dataclasses import dataclass
from typing import Any
from torch import optim, nn
from copy import deepcopy
from denspp.offline.dnn.model_library import ModelLibrary


@dataclass
class ConfigPytorch:
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
    def get_model_overview(print_overview: bool=False, index: str='') -> None:
        """Function for getting an overview of existing models inside library"""
        models_bib = ModelLibrary().get_registry()
        models_bib.get_library_overview(index, do_print=print_overview)

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
        """Loading the optimizer function"""
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
            raise NotImplementedError("Please select one model above and type-in the name into yaml file")
        else:
            if models_bib.check_module_available(self.model_name):
                used_model = deepcopy(models_bib.build(self.model_name, *args, **kwargs))
                return used_model
            else:
                ovr = models_bib.get_library_overview(do_print=True)
                raise NotImplementedError(f"Model is not available - Please check again!")



DefaultSettingsTrainMSE = ConfigPytorch(
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
DefaultSettingsTrainCE = ConfigPytorch(
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
