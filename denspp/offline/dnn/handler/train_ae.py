from copy import deepcopy
from datetime import date
from denspp.offline.yaml_handler import YamlConfigHandler
from denspp.offline.dnn.dnn_handler import ConfigMLPipeline
from denspp.offline.dnn.pytorch_config_data import ConfigDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainMSE
from denspp.offline.dnn.pytorch_pipeline import do_train_autoencoder
from denspp.offline.dnn.plots.plot_dnn import results_training
from denspp.offline.dnn.dataset.autoencoder import prepare_training


def do_train_neural_autoencoder(settings: ConfigMLPipeline, yaml_name_index='Config_AE',
                                model_default_name='', used_dataset_name='quiroga') -> [dict, dict]:
    """Training routine for Autoencoders in Neural Applications (Spike Frames)
    Args:
        settings:               Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:        Index of yaml file name
        model_default_name:     Optional name for the model to load
        used_dataset_name:      Default name of the dataset for training [default: quiroga]
    Returns:
        Dictionaries with results from training [metrics, validation data]
    """
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = used_dataset_name
    yaml_data = YamlConfigHandler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(ConfigDataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainMSE)
    default_train.model_name = model_default_name
    yaml_nn = YamlConfigHandler(default_train, settings.get_path2config, f'{yaml_name_index}_Training')
    config_train = yaml_nn.get_class(ConfigPytorch)
    del default_train, yaml_nn

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(settings=config_data, do_classification=False,
                               mode_train_ae=settings.autoencoder_mode, noise_std=settings.autoencoder_noise_std)
    if settings.autoencoder_feat_size:
        used_model = config_train.get_model(output_size=settings.autoencoder_feat_size)
    else:
        used_model = config_train.get_model()

    metrics, data_result, path2folder = do_train_autoencoder(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model, calc_custom_metrics=['dsnr_all', 'ptq_loss'],
        save_vhdl=True, path4vhdl=f'vhdl/run_{date.today()}'
    )

    # --- Plotting
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
    from offline.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_neural_autoencoder(DefaultSettings_MLPipe)
