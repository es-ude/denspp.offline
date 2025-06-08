from copy import deepcopy

from denspp.offline.yaml_handler import YamlHandler
from denspp.offline.dnn.dnn_handler import ConfigMLPipeline
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE
from denspp.offline.dnn.pytorch_pipeline import train_classifier_template, train_autoencoder_template
from denspp.offline.dnn.dataset.torch_datasets import prepare_training
from denspp.offline.dnn.plots.plot_dnn import plot_mnist_graphs


def do_train_torchvision_cl(class_dataset, settings: ConfigMLPipeline, dataset_type: str) -> None:
    """Training routine for classifying MNIST
    Args:
        class_dataset:      Class of custom-made SettingsDataset from src_dnn/call_dataset.py
        settings:           Handler for configuring the routine selection for train deep neural networks
        dataset_type:       Selected dataset type from Torchvision [MNIST, Fashion, Cifar-10, Cifar-100]
    Returns:
        None
    """
    yaml_name_index = f'Config_{dataset_type.upper()}'

    # --- Loading the YAML files: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = dataset_type.upper()
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = f'{dataset_type.lower()}_mlp_cl_v1'
    config_train = YamlHandler(
        template=default_train, path=settings.get_path2config, file_name=f'{yaml_name_index}_TrainCL').get_class(ConfigPytorch)

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        do_classification=True
    )
    used_model = config_train.get_model()
    train_classifier_template(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model
    )


def do_train_torchvision_ae(class_dataset, settings: ConfigMLPipeline, dataset_type: str) -> None:
    """Training routine for training an autoencoder
    Args:
        class_dataset:      Class of custom-made SettingsDataset from src_dnn/call_dataset.py
        settings:           Handler for configuring the routine selection for train deep neural networks
        dataset_type:       Selected dataset type from Torchvision [MNIST, Fashion, Cifar-10, Cifar-100]
    Returns:
        None
    """
    yaml_name_index =  f'Config_{dataset_type.upper()}'

    # --- Loading the YAML file: Dataset
    default_data = DefaultSettingsDataset
    default_data.data_file_name = f'{dataset_type.upper()}'
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Model training
    default_train = DefaultSettingsTrainMSE
    default_train.model_name = f'{dataset_type.lower()}_mlp_ae_v1'
    config_train = YamlHandler(
        template=default_train,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainAE'
    ).get_class(ConfigPytorch)
    del default_data, default_train

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        do_classification=False
    )
    used_model = config_train.get_model()
    metrics, data_result, path2save = train_autoencoder_template(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model
    )

    # --- Additional plots
    if settings.do_plot:
        plot_mnist_graphs(data_result['input'], data_result['valid_clus'], "_input", path2save=path2save)
        plot_mnist_graphs(data_result['pred'], data_result['valid_clus'], "_predicted", path2save=path2save,
                          show_plot=settings.do_block)
