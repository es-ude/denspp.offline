from copy import deepcopy

from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import ConfigMLPipeline
from package.dnn.pytorch_config_data import ConfigDataset, DefaultSettingsDataset
from package.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_classifier, do_train_autoencoder
from package.dnn.dataset.mnist import prepare_training
from package.dnn.plots.plot_dnn import plot_mnist_graphs


def do_train_mnist_cl(settings: ConfigMLPipeline, yaml_name_index='Config_MNIST', custom_metrics=()) -> None:
    """Training routine for classifying neural activations
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:    Index of yaml file name for saving and configure configuration
        custom_metrics:     List with metrics for calculation during validation phase ['accuracy', 'precision', 'recall', 'fbeta']
    Returns:
        None
    """
    # --- Loading the YAML files: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_path = 'data'
    default_data.data_file_name = 'MNIST'
    yaml_data = yaml_config_handler(default_data, path2yaml=settings.get_path2config, yaml_name=f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(ConfigDataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = 'mnist_mlp_cl_v1'
    yaml_train = yaml_config_handler(default_train, path2yaml=settings.get_path2config, yaml_name=f'{yaml_name_index}_TrainCL')
    config_train = yaml_train.get_class(ConfigPytorch)

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(config_data, True)
    used_model = config_train.get_model()
    do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model, calc_custom_metrics=custom_metrics
    )


def do_train_mnist_ae(settings: ConfigMLPipeline, yaml_name_index='Config_MNIST') -> None:
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
    default_data.data_file_name = 'MNIST'
    yaml_data = yaml_config_handler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(ConfigDataset)

    # --- Loading the YAML file: Model training
    default_train = DefaultSettingsTrainMSE
    default_train.model_name = 'mnist_mlp_ae_v1'
    yaml_train = yaml_config_handler(default_train, settings.get_path2config, f'{yaml_name_index}_TrainAE')
    config_train = yaml_train.get_class(ConfigPytorch)

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(config_data, False)
    used_model = config_train.get_model()
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
    do_train_mnist_cl(set0)

    set0.do_plot = True
    do_train_mnist_ae(set0)
