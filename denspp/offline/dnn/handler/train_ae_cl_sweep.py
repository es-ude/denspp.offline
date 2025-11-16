import os
import numpy as np
from shutil import rmtree
from copy import deepcopy
from datetime import datetime
from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.dnn.dnn_handler import SettingsMLPipeline
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE
from denspp.offline.dnn import train_classifier_routine, train_autoencoder_routine
from denspp.offline.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
from denspp.offline.dnn.dataset.autoencoder_class import prepare_training as get_dataset_cl


def do_train_ae_cl_sweep(class_dataset, settings: SettingsMLPipeline,
                         feat_layer_start: int, feat_layer_inc: int, feat_layer_stop: int,
                         num_epochs_trial: int=50, yaml_name_index: str= 'Config_AECL_Sweep',
                         model_ae_default_name: str='', model_cl_default_name: str='',
                         used_dataset_name:str='quiroga') -> str:
    """Training routine for Autoencoders and Classification after Encoder (Sweep)
    :param class_dataset:           Class of custom-made SettingsDataset from src_dnn/call_dataset.py
    :param settings:                Handler for configuring the routine selection for train deep neural networks
    :param feat_layer_start:        Increasing value for feature layer
    :param feat_layer_inc:          Increasing value for feature layer
    :param feat_layer_stop:         Increasing value for feature layer
    :param num_epochs_trial:        Number of epochs of each run
    :param yaml_name_index:         Index of yaml file name
    :param model_ae_default_name:   Default name for autoencoder model
    :param model_cl_default_name:   Default name for classifier model
    :param used_dataset_name:       Default dataset name used in training
    :return:                        String with path in which the data is saved
    """
    # ------------ STEP #0: Loading YAML files
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = used_dataset_name
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Autoencoder Model Load and building
    default_ae = deepcopy(DefaultSettingsTrainMSE)
    default_ae.model_name = model_ae_default_name
    default_ae.num_epochs = num_epochs_trial
    config_train_ae = YamlHandler(
        template=default_ae,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainAE'
    ).get_class(ConfigPytorch)

    # --- Loading the YAML file: Classifier Model Load and building
    default_cl = deepcopy(DefaultSettingsTrainCE)
    default_cl.model_name = model_cl_default_name
    default_cl.num_epochs = num_epochs_trial
    config_train_cl = YamlHandler(
        template=default_cl,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainCL'
    ).get_class(ConfigPytorch)
    del default_data, default_ae, default_cl

    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_foldername = f'{time_now}_{config_train_ae.model_name}_sweep'
    path2save = os.path.join(config_data.get_path2folder_project, 'runs', sweep_foldername)
    if os.path.exists(path2save):
        rmtree(path2save)

    num_clusters = 0
    metrics_runs = dict()
    sweep_val = [idx for idx in range(feat_layer_start, feat_layer_stop, feat_layer_inc)]
    sweep_val.append(feat_layer_stop)
    for idx, feat_size in enumerate(sweep_val):
        path2save_base = f"{path2save}/sweep_{idx:02d}_size{feat_size}"
        # ----------- Step #1: TRAINING AUTOENCODER
        used_dataset_ae = get_dataset_ae(
            rawdata=class_dataset(settings=config_data).load_dataset(),
            mode_train_ae=settings.autoencoder_mode,
            noise_std=settings.autoencoder_noise_std,
            do_classification=False
        )
        used_model_ae = config_train_ae.get_model(input_size=class_dataset[0]['in'].size, output_size=feat_size)
        metrics_ae, valid_data_ae, path2folder = train_autoencoder_routine(
            config_ml=settings,
            config_train=config_train_ae,
            config_data=config_data,
            path2save=path2save_base,
            used_dataset=used_dataset_ae,
            used_model=used_model_ae
        )
        del used_dataset_ae, used_model_ae

        # ----------- Step #2: TRAINING CLASSIFIER
        used_dataset_cl = get_dataset_cl(
            rawdata=class_dataset(settings=config_data).load_dataset(),
            path2model=path2folder
        )
        used_model_cl = config_train_cl.get_model(input_size=feat_size, output_size=used_dataset_cl.get_cluster_num)
        metrics_cl = train_classifier_routine(
            config_ml=settings,
            config_train=config_train_cl,
            config_data=config_data,
            path2save=path2save_base,
            used_dataset=used_dataset_cl,
            used_model=used_model_cl
        )
        if idx == 0:
            num_clusters = used_dataset_cl.get_cluster_num

        del used_dataset_cl, used_model_cl
        metrics_runs.update({f"feat_{feat_size:03d}_ae": metrics_ae, f"feat_{feat_size:03d}_cl": metrics_cl})

    metrics_runs.update({'num_clusters': num_clusters})
    # ----------- Step #3: Output results
    np.save(f'{path2save}/results_sweep.npy', metrics_runs, allow_pickle=True)
    return path2save
