from copy import deepcopy
from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.dnn.dnn_handler import SettingsMLPipeline
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainCE
from denspp.offline.dnn.pytorch_pipeline import train_classifier_template
from denspp.offline.dnn.dataset.classifier import prepare_training


def do_train_classifiers(class_dataset, settings: SettingsMLPipeline,
                         yaml_name_index: str='Config_Neural', used_dataset_name: str='quiroga', used_model_name: str='') -> str:
    """Training routine for Classification DL models
    :param class_dataset:           Class of custom-made SettingsDataset from src_dnn/call_dataset.py
    :param settings:                Handler for configuring the routine selection for train deep neural networks
    :param yaml_name_index:         Index of yaml file name
    :param used_dataset_name:       Used dataset name
    :param used_model_name:         Used model for DNN training
    :return:                        String with path to folder in which results are saved
    """
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = used_dataset_name
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = used_model_name
    config_train = YamlHandler(
        template=default_train,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainCL'
    ).get_class(ConfigPytorch)
    del default_train, default_data

    # --- Loading Data, Build Model and Do Inference
    dataset = prepare_training(
        rawdata=class_dataset(settings=config_data).load_dataset()
    )
    used_model = config_train.get_model(input_size=dataset[0]['in'].size, output_size=dataset.get_cluster_num)
    _, _, path2folder = train_classifier_template(
        config_ml=settings,
        config_data=config_data,
        config_train=config_train,
        used_dataset=dataset,
        used_model=used_model
    )
    return path2folder
