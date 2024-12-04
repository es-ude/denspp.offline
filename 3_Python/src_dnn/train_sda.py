from copy import deepcopy
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import Config_Dataset, DefaultSettingsDataset, Config_PyTorch, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_classifier

from package.dnn.template.dataset.spike_detection import prepare_training
import package.dnn.template.models.spike_detection as models


def dnn_train_sda(settings: Config_ML_Pipeline, sda_threshold=4) -> None:
    """Training routine for Spike Detection
    Args:
        settings:       Handler for configuring the routine selection for train deep neural networks
        sda_threshold:  Threshold value for identifying a spike event
    Return:
        None
    """
    # --- Loading the YAML file: Dataset
    yaml_data = yaml_config_handler(DefaultSettingsDataset, yaml_name='Config_SDA_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = models.dnn_sda_v1.__name__
    yaml_train = yaml_config_handler(default_train, 'config', 'Config_SDA_Train')
    config_train = yaml_train.get_class(Config_PyTorch)

    # --- Loading Data, Build Model and Do Training
    used_dataset = prepare_training(config_data, sda_threshold)
    used_model = models.models_available.build_model(config_train.model_name)
    do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=used_dataset, used_model=used_model
    )


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    dnn_train_sda(DefaultSettings_MLPipe)
