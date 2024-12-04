from copy import deepcopy
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import Config_Dataset, DefaultSettingsDataset, Config_PyTorch, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_classifier, get_model_attributes

import src_dnn.models.decoding_utah as models
from src_dnn.dataset.decoding_utah import prepare_training


def do_train_decoder_utah(config_ml: Config_ML_Pipeline, length_window_ms=500) -> None:
    """Training routine for Neural Decoding of recordings from Utah array (KlaesLab)
    Args:
        config_ml:          Handler for configuring the routine selection for train deep neural networks
        length_window_ms:   Size of the time window for segmenting the tick interval into firing events
    Return:
        None
    """
    common_yaml_name = 'Config_Utah'
    # --- Loading the YAML file: Dataset
    yaml_data = yaml_config_handler(DefaultSettingsDataset,
                                    path2yaml='config', yaml_name=f'{common_yaml_name}_Data')
    config_data = yaml_data.get_class(Config_Dataset)
    del yaml_data

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = get_model_attributes(models, '_dec_v')
    yaml_train = yaml_config_handler(default_train, path2yaml='config',
                                     yaml_name=f'{common_yaml_name}_Train')
    config_train = yaml_train.get_class(Config_PyTorch)
    del default_train, yaml_train

    # --- Loading Data, Build Model and Do Training
    used_dataset = prepare_training(config_data, length_window_ms, use_cluster=False)
    used_model = models.models_available.build_model(config_train.model_name)
    do_train_classifier(
        config_ml, config_train, config_data, used_dataset, used_model
    )


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_decoder_utah(DefaultSettings_MLPipe)
