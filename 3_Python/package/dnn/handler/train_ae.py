from copy import deepcopy
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsTrainMSE,
                                           Config_Dataset, DefaultSettingsDataset)
from package.dnn.pytorch_pipeline import do_train_autoencoder

from package.plot.plot_dnn import results_training
from package.dnn.template.dataset.autoencoder import prepare_training
import package.dnn.template.models.autoencoder_dnn as models


def do_train_neural_autoencoder(settings: Config_ML_Pipeline, add_noise_cluster=False,
                                yaml_name_index='Config_AE') -> [dict, dict]:
    """Training routine for Autoencoders in Neural Applications (Spike Frames)
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        add_noise_cluster:  Decision for adding noise cluster activity
        yaml_name_index:    Index of yaml file name
    Returns:
        Dictionaries with results from training [metrics, validation data]
    """
    # --- Loading the YAML file: Dataset
    yaml_data = yaml_config_handler(DefaultSettingsDataset, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainMSE)
    default_train.model_name = models.dnn_ae_v1.__name__
    yaml_nn = yaml_config_handler(default_train, settings.get_path2config, f'{yaml_name_index}_Training')
    config_train = yaml_nn.get_class(Config_PyTorch)
    del default_train, yaml_nn

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(settings=config_data, do_classification=False,
                               mode_train_ae=settings.autoencoder_mode, noise_std=settings.autoencoder_noise_std,
                               add_noise_cluster=add_noise_cluster,
                               use_median_for_mean=True)
    if settings.autoencoder_feat_size:
        used_model = models.models_available.build_model(config_train.model_name, output_size=settings.autoencoder_feat_size)
    else:
        used_model = models.models_available.build_model(config_train.model_name)

    metrics, data_result, path2folder = do_train_autoencoder(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model, calc_custom_metrics=['dsnr_all']
    )

    if settings.do_plot:
        used_first_fold = [key for key in metrics.keys()][0]
        results_training(
            path=path2folder, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
            yin=data_result['input'], ypred=data_result['pred'], ymean=dataset.get_mean_waveforms,
            yclus=data_result['valid_clus'], snr=metrics[used_first_fold]['dsnr_all'],
            show_plot=settings.do_block
        )
    return metrics, data_result


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_neural_autoencoder(DefaultSettings_MLPipe)
