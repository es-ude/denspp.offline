from copy import deepcopy
from denspp.offline.yaml_handler import YamlConfigHandler
from denspp.offline.dnn.dnn_handler import ConfigMLPipeline
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE
from denspp.offline.dnn.pytorch_pipeline import do_train_autoencoder, do_train_classifier
from denspp.offline.dnn.plots.plot_dnn import results_training
from denspp.offline.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
from denspp.offline.dnn.dataset.autoencoder_class import prepare_training as get_dataset_cl


def do_train_ae_classifier(class_dataset, settings: ConfigMLPipeline,
                           yaml_name_index: str= 'Config_ACL',
                           model_ae_default_name: str='', model_cl_default_name: str='',
                           used_dataset_name:str='quiroga') -> dict:
    """Training routine for Autoencoders and Classifier with Encoder after Autoencoder-Training
    Args:
        class_dataset:      Class of custom-made SettingsDataset from src_dnn/call_dataset.py
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:    Index of yaml file name
        model_ae_default_name:  Default name of the autoencoder model
        model_cl_default_name:  Default name of the classifier model
        used_dataset_name:  Default name of the dataset for training [default: quiroga]
    Returns:
        Dictionary with metric results from Autoencoder and Classifier Training
    """
    # --- Processing Step #0: Loading YAML files
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = used_dataset_name
    yaml_data = YamlConfigHandler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(SettingsDataset)
    del yaml_data

    # --- Loading the YAML file: Autoencoder Model training
    default_train_ae = deepcopy(DefaultSettingsTrainMSE)
    default_train_ae.model_name = model_ae_default_name
    yaml_nn0 = YamlConfigHandler(default_train_ae, settings.get_path2config, f'{yaml_name_index}_TrainAE')
    config_train_ae = yaml_nn0.get_class(ConfigPytorch)
    del default_train_ae, yaml_nn0

    # --- Loading the YAML file: Classifier Model training
    default_train_cl = deepcopy(DefaultSettingsTrainCE)
    default_train_cl.model_name = model_cl_default_name
    yaml_nn1 = YamlConfigHandler(default_train_cl, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train_cl = yaml_nn1.get_class(ConfigPytorch)
    del default_train_cl, yaml_nn1

    # --- Processing Step #1.1: Loading dataset and Build Model
    dataset = get_dataset_ae(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        do_classification=False,
        mode_train_ae=settings.autoencoder_mode,
        noise_std=settings.autoencoder_noise_std
    )
    if settings.autoencoder_feat_size:
        used_model_ae = config_train_ae.get_model(output_size=settings.autoencoder_feat_size)
    else:
        used_model_ae = config_train_ae.get_model()

    print("\n# ----------- Step #1: TRAINING AUTOENCODER")
    # --- Processing Step #1.2: Train Autoencoder and Plot Results
    metrics_ae, valid_data_ae, path2folder = do_train_autoencoder(
        config_ml=settings, config_data=config_data, config_train=config_train_ae,
        used_dataset=dataset, used_model=used_model_ae, path2save='', calc_custom_metrics=['dsnr_all']
    )
    if settings.do_plot:
        used_first_fold = [key for key in metrics_ae.keys()][0]
        results_training(
            path=path2folder, cl_dict=dataset.get_dictionary, feat=valid_data_ae['feat'],
            yin=valid_data_ae['input'], ypred=valid_data_ae['pred'], ymean=dataset.get_mean_waveforms,
            yclus=valid_data_ae['valid_clus'], snr=metrics_ae[used_first_fold]['dsnr_all']
        )
    del dataset

    print("\n# ----------- Step #2: TRAINING CLASSIFIER")
    # --- Processing Step #2.1: Loading dataset and Build Model
    dataset = get_dataset_cl(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        path2model=path2folder,
        print_state=True
    )
    num_feat = dataset[0]['in'].shape[0] if not settings.autoencoder_feat_size else settings.autoencoder_feat_size
    used_model_cl = config_train_cl.get_model(input_size=num_feat, output_size=dataset.get_cluster_num)

    # --- Processing Step #1.2: Train Classifier
    metrics_cl, valid_data_cl, _ = do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train_cl,
        used_dataset=dataset, used_model=used_model_cl, path2save=path2folder
    )
    return {"path2model_ae": path2folder, 'metric_ae': metrics_ae, 'metrics_cl': metrics_cl}
