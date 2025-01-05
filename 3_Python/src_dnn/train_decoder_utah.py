from copy import deepcopy
from package.yaml_handler import YamlConfigHandler
from package.dnn.dnn_handler import ConfigMLPipeline
from package.dnn.pytorch_config_data import ConfigDataset, DefaultSettingsDataset
from package.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_classifier
from src_dnn.dataset.decoding_utah import prepare_training


def do_train_decoder_utah(config_ml: ConfigMLPipeline, length_window_ms=500,
                          yaml_name_index='Config_Utah') -> None:
    """Training routine for Neural Decoding of recordings from Utah array (KlaesLab)
    Args:
        config_ml:          Handler for configuring the routine selection for train deep neural networks
        length_window_ms:   Size of the time window for segmenting the tick interval into firing events
        yaml_name_index:    Index of yaml file name
    Return:
        None
    """
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = ''
    yaml_data = YamlConfigHandler(DefaultSettingsDataset, path2yaml='config', yaml_name=f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(ConfigDataset)
    del yaml_data

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = ''
    yaml_train = YamlConfigHandler(default_train, path2yaml='config', yaml_name=f'{yaml_name_index}_Train')
    config_train = yaml_train.get_class(ConfigPytorch)
    del default_train, yaml_train

    # --- Loading Data, Build Model and Do Training
    used_dataset = prepare_training(config_data, length_window_ms, use_cluster=False)
    used_model = config_train.get_model()
    do_train_classifier(
        config_ml, config_train, config_data, used_dataset, used_model
    )


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_decoder_utah(DefaultSettings_MLPipe)
