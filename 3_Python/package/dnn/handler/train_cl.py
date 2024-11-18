from copy import deepcopy
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsDataset,
                                           Config_Dataset, DefaultSettingsTrainCE)
from package.dnn.pytorch_pipeline import do_train_classifier

from package.dnn.template.dataset.autoencoder import prepare_training
import package.dnn.template.models.spike_classifier as models


def do_train_neural_spike_classification(settings: Config_ML_Pipeline, yaml_name_index='Config_Neural') -> None:
    """Training routine for Classification DL models
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:    Index of yaml file name
    Returns:
        None
    """
    # --- Loading the YAML file: Dataset
    yaml_data = yaml_config_handler(DefaultSettingsDataset, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = models.spike_cl_v1.__name__
    yaml_train = yaml_config_handler(default_train, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train = yaml_train.get_class(Config_PyTorch)

    # --- Loading Data, Build Model and Do Inference
    dataset = prepare_training(config_data, do_classification=True)
    used_model = models.models_available.build_model(config_train.model_name)
    do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model
    )


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_neural_spike_classification(DefaultSettings_MLPipe)
