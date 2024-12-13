import os
import numpy as np
from shutil import rmtree
from copy import deepcopy
from datetime import datetime
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline, DefaultSettings_MLPipe
from package.dnn.pytorch_config_data import Config_Dataset, DefaultSettingsDataset
from package.dnn.pytorch_config_model import Config_PyTorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_autoencoder, do_train_classifier
from package.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.dataset.autoencoder_class import prepare_training as get_dataset_cl


def do_train_ae_cl_sweep(settings: Config_ML_Pipeline,
                         feat_layer_start: int, feat_layer_inc: int, feat_layer_stop: int,
                         num_epochs_trial=50,
                         yaml_name_index='Config_AECL_Sweep') -> str:
    """Training routine for Autoencoders and Classification after Encoder (Sweep)
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        feat_layer_start:   Increasing value for feature layer
        feat_layer_inc:     Increasing value for feature layer
        feat_layer_stop:    Increasing value for feature layer
        add_noise_cluster:  Adding noise cluster to dataset [Default: False]
        num_epochs_trial:   Number of epochs of each run
        yaml_name_index:    Index of yaml file name
    Return:
        String with path in which the data is saved
    """
    # ------------ STEP #0: Loading YAML files
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    yaml_data = yaml_config_handler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Autoencoder Model Load and building
    default_ae = deepcopy(DefaultSettingsTrainMSE)
    default_ae.model_name = ''
    default_ae.num_epochs = num_epochs_trial
    yaml_train = yaml_config_handler(default_ae, settings.get_path2config, f'{yaml_name_index}_TrainAE')
    config_train_ae = yaml_train.get_class(Config_PyTorch)
    del yaml_train, default_ae

    # --- Loading the YAML file: Classifier Model Load and building
    default_cl = deepcopy(DefaultSettingsTrainCE)
    default_cl.model_name = ''
    default_cl.num_epochs = num_epochs_trial
    yaml_train = yaml_config_handler(default_cl, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train_cl = yaml_train.get_class(Config_PyTorch)
    del yaml_train, default_cl

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
            settings=config_data,
            mode_train_ae=settings.autoencoder_mode,
            noise_std=settings.autoencoder_noise_std,
            do_classification=False
        )
        used_model_ae = config_train_ae.get_model(output_size=feat_size)
        metrics_ae, valid_data_ae, path2folder = do_train_autoencoder(
            config_ml=settings,
            config_train=config_train_ae,
            config_data=config_data,
            path2save=path2save_base,
            used_dataset=used_dataset_ae,
            used_model=used_model_ae,
            calc_custom_metrics=['snr_in', 'snr_in_cl', 'dsnr_all', 'dsnr_cl']
        )
        del used_dataset_ae, used_model_ae

        # ----------- Step #2: TRAINING CLASSIFIER
        used_dataset_cl = get_dataset_cl(
            settings=config_data,
            path2model=path2folder
        )
        used_model_cl = config_train_cl.get_model(input_size=feat_size, output_size=used_dataset_cl.get_cluster_num)
        metrics_cl = do_train_classifier(
            config_ml=settings,
            config_train=config_train_cl,
            config_data=config_data,
            path2save=path2save_base,
            used_dataset=used_dataset_cl,
            used_model=used_model_cl,
            calc_custom_metrics=['precision']
        )
        if idx == 0:
            num_clusters = used_dataset_cl.get_cluster_num

        del used_dataset_cl, used_model_cl
        metrics_runs.update({f"feat_{feat_size:03d}_ae": metrics_ae, f"feat_{feat_size:03d}_cl": metrics_cl})

    metrics_runs.update({'num_clusters': num_clusters})
    # ----------- Step #3: Output results
    np.save(f'{path2save}/results_sweep.npy', metrics_runs, allow_pickle=True)
    return path2save


if __name__ == "__main__":
    from package.dnn.dnn_handler import Config_ML_Pipeline
    from package.plot.plot_ae_cl_sweep import extract_data_from_files, plot_common_loss, plot_common_params, plot_architecture_metrics_isolated

    yaml_handler = yaml_config_handler(DefaultSettings_MLPipe, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(Config_ML_Pipeline)
    dnn_handler.do_plot = True
    dnn_handler.do_block = False

    # --- Step #1: Run results
    print("========================================\n Sweep Run for Training Autoencoder + Classification System\n")
    path2save = do_train_ae_cl_sweep(dnn_handler, 1, 1, 32)

    # --- Step #2: Plot results
    print("===========================================\n Printing results and plot results\n")
    data = extract_data_from_files(path2save)
    plot_common_loss(data, path2save=path2save)
    plot_common_params(data, path2save=path2save)
    plot_architecture_metrics_isolated(data, show_plots=True, path2save=path2save)
    print("\n.done")
