from copy import deepcopy
from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.dnn.dnn_handler import SettingsMLPipeline
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainMSE
from denspp.offline.dnn import train_autoencoder_routine
from denspp.offline.dnn.plots.plot_dnn import results_training
from denspp.offline.dnn.dataset.autoencoder import prepare_training


def do_train_autoencoder(class_dataset, settings: SettingsMLPipeline, yaml_name_index: str= 'Config_AE',
                         used_dataset_name: str='quiroga', model_default_name: str='') -> tuple[dict, dict]:
    """Training routine for Autoencoders (e.g. in neural Applications for Spike Frames)
    Args:
        class_dataset:          Class of custom-made SettingsDataset from src_dnn/call_dataset.py
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
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainMSE)
    default_train.model_name = model_default_name
    default_train.custom_metrics = ['dsnr_all']
    config_train = YamlHandler(
        template=default_train,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainAE'
    ).get_class(ConfigPytorch)
    del default_train, default_data

    # --- Loading Data, Build Model and Do Training
    dataset = prepare_training(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        do_classification=False,
        mode_train_ae=settings.autoencoder_mode,
        noise_std=settings.autoencoder_noise_std,
        print_state=True
    )
    if settings.autoencoder_feat_size:
        used_model = config_train.get_model(input_size=dataset[0]['in'].size, output_size=settings.autoencoder_feat_size)
    else:
        used_model = config_train.get_model(input_size=dataset[0]['in'].size, output_size=dataset[0]['in'].size)

    metrics, data_result, path2folder = train_autoencoder_routine(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=used_model
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
