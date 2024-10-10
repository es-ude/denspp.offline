from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsTrainCE, DefaultSettingsTrainMSE,
                                           Config_Dataset, DefaultSettingsDataset)
from package.dnn.pytorch_pipeline import do_train_classifier, do_train_autoencoder

from package.dnn.template.dataset.mnist import prepare_training
import package.dnn.template.models.mnist as models
from package.plot.plot_dnn import plot_mnist_graphs


def do_train_cl(settings: Config_ML_Pipeline, yaml_name_index='Config_MNIST') -> None:
    """Training routine for classifying neural activations
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:
    Returns:
        None
    """
    # --- Loading the YAML files: Dataset
    default_data = DefaultSettingsDataset
    default_data.data_path = 'data'
    yaml_data = yaml_config_handler(default_data,
                                    path2yaml=settings.get_path2config, yaml_name=f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Model training
    default_train = DefaultSettingsTrainCE
    default_train.model_name = models.mnist_mlp_cl_v1.__name__
    yaml_train = yaml_config_handler(default_train,
                                     path2yaml=settings.get_path2config, yaml_name=f'{yaml_name_index}_TrainCL')
    config_train = yaml_train.get_class(Config_PyTorch)

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(config_data, True)
    used_model = models.models_available.build_model(config_train.model_name)
    do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model
    )


def do_train_ae(settings: Config_ML_Pipeline, yaml_name_index='Config_MNIST') -> None:
    """Training routine for training an autoencoder
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:    Index of yaml file name
    Returns:
        None
    """
    # --- Loading the YAML file: Dataset
    default_data = DefaultSettingsDataset
    default_data.data_path = 'data'
    yaml_data = yaml_config_handler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Model training
    default_train = DefaultSettingsTrainMSE
    default_train.model_name = models.mnist_mlp_ae_v1.__name__
    yaml_train = yaml_config_handler(default_train, settings.get_path2config, f'{yaml_name_index}_TrainAE')
    config_train = yaml_train.get_class(Config_PyTorch)

    # --- Loading Data, Build Model and Do Training
    print(config_data.get_path2data)

    dataset = prepare_training(config_data, False)
    used_model = models.models_available.build_model(config_train.model_name)
    metrics, data_result, path2save = do_train_autoencoder(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model
    )

    # --- Additional plots
    if settings.do_plot:
        plot_mnist_graphs(data_result['input'], data_result['valid_clus'], "_input", path2save=path2save)
        plot_mnist_graphs(data_result['pred'], data_result['valid_clus'], "_predicted", path2save=path2save,
                          show_plot=settings.do_block)


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe

    set0 = DefaultSettings_MLPipe
    set0.do_plot = False
    do_train_cl(set0)

    set0.do_plot = True
    do_train_ae(set0)
