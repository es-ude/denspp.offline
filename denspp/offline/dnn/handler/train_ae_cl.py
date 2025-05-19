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
    config_data = YamlConfigHandler(
        yaml_template=default_data,
        path2yaml=settings.get_path2config,
        yaml_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Autoencoder Model training
    default_train_ae = deepcopy(DefaultSettingsTrainMSE)
    default_train_ae.model_name = model_ae_default_name
    config_train_ae = YamlConfigHandler(
        yaml_template=default_train_ae,
        path2yaml=settings.get_path2config,
        yaml_name=f'{yaml_name_index}_TrainAE'
    ).get_class(ConfigPytorch)

    # --- Loading the YAML file: Classifier Model training
    default_train_cl = deepcopy(DefaultSettingsTrainCE)
    default_train_cl.model_name = model_cl_default_name
    config_train_cl = YamlConfigHandler(
        yaml_template=default_train_cl,
        path2yaml=settings.get_path2config,
        yaml_name=f'{yaml_name_index}_TrainCL'
    ).get_class(ConfigPytorch)
    del default_train_cl, default_train_ae, default_data

    # --- Processing Step #1.1: Loading dataset and Build Model
    dataset_ae = get_dataset_ae(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        do_classification=False,
        mode_train_ae=settings.autoencoder_mode,
        noise_std=settings.autoencoder_noise_std
    )
    if settings.autoencoder_feat_size:
        used_model_ae = config_train_ae.get_model(input_size=dataset_ae[0]['in'].size, output_size=settings.autoencoder_feat_size)
    else:
        used_model_ae = config_train_ae.get_model(input_size=dataset_ae[0]['in'].size)

    print("\n# ----------- Step #1: TRAINING AUTOENCODER")
    # --- Processing Step #1.2: Train Autoencoder and Plot Results
    metrics_ae, valid_data_ae, path2folder = do_train_autoencoder(
        config_ml=settings, config_data=config_data, config_train=config_train_ae,
        used_dataset=dataset_ae, used_model=used_model_ae, path2save=''
    )
    if settings.do_plot:
        used_first_fold = [key for key in metrics_ae.keys()][0]
        results_training(
            path=path2folder, cl_dict=dataset_ae.get_dictionary, feat=valid_data_ae['feat'],
            yin=valid_data_ae['input'], ypred=valid_data_ae['pred'], ymean=dataset_ae.get_mean_waveforms,
            yclus=valid_data_ae['valid_clus'], snr=metrics_ae[used_first_fold]['dsnr_all']
        )
    del dataset_ae

    print("\n# ----------- Step #2: TRAINING CLASSIFIER")
    # --- Processing Step #2.1: Loading dataset and Build Model
    dataset_cl = get_dataset_cl(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        path2model=path2folder,
        print_state=True
    )
    num_feat = dataset_cl[0]['in'].shape[0] if not settings.autoencoder_feat_size else settings.autoencoder_feat_size
    used_model_cl = config_train_cl.get_model(input_size=num_feat, output_size=dataset_cl.get_cluster_num)

    # --- Processing Step #1.2: Train Classifier
    metrics_cl, valid_data_cl, _ = do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train_cl,
        used_dataset=dataset_cl, used_model=used_model_cl, path2save=path2folder
    )
    del dataset_cl
    return {"path2model_ae": path2folder, 'metric_ae': metrics_ae, 'metrics_cl': metrics_cl}
