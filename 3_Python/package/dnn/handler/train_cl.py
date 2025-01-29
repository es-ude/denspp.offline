from copy import deepcopy
from package.yaml_handler import YamlConfigHandler
from package.dnn.dnn_handler import ConfigMLPipeline
from package.dnn.pytorch_config_data import ConfigDataset, DefaultSettingsDataset
from package.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_classifier
from package.dnn.dataset.classifier import prepare_training


def do_train_spike_class(settings: ConfigMLPipeline, yaml_name_index='Config_Neural',
                         used_dataset_name='', used_model_name='') -> str:
    """Training routine for Classification DL models
    Args:
        settings:               Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:        Index of yaml file name
        used_dataset_name:      Used dataset name
        used_model_name:        Used model for DNN training
    Returns:
        String with path to folder in which results are saved
    """
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = used_dataset_name
    yaml_data = YamlConfigHandler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(ConfigDataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = used_model_name
    yaml_train = YamlConfigHandler(default_train, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train = yaml_train.get_class(ConfigPytorch)

    # --- Loading Data, Build Model and Do Inference
    dataset = prepare_training(config_data)
    used_model = config_train.get_model(input_size=dataset[0]['in'].size, output_size=dataset.get_cluster_num)
    _, _, path2folder = do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model
    )
    return path2folder


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_spike_class(DefaultSettings_MLPipe)
