from copy import deepcopy
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_config_data import Config_Dataset, DefaultSettingsDataset
from package.dnn.pytorch_config_model import Config_PyTorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_autoencoder, do_train_classifier
from package.plot.plot_dnn import results_training
from package.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.dataset.autoencoder_class import prepare_training as get_dataset_cl


def do_train_ae_classifier(settings: Config_ML_Pipeline, yaml_name_index: str='Config_ACL',
                           model_ae_default_name: str='', model_cl_default_name: str='', used_dataset_name:str='quiroga',
                           add_noise_cluster: bool=False) -> dict:
    """Training routine for Autoencoders and Classifier with Encoder after Autoencoder-Training
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:    Index of yaml file name
        model_ae_default_name:  Default name of the autoencoder model
        model_cl_default_name:  Default name of the classifier model
        used_dataset_name:  Default name of the dataset for training [default: quiroga]
        add_noise_cluster:  Adding noise cluster to dataset [Default: False]
    Returns:
        Dictionary with metric results from Autoencoder and Classifier Training
    """
    # --- Processing Step #0: Loading YAML files
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = used_dataset_name
    yaml_data = yaml_config_handler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del yaml_data

    # --- Loading the YAML file: Autoencoder Model training
    default_train_ae = deepcopy(DefaultSettingsTrainMSE)
    default_train_ae.model_name = model_ae_default_name
    yaml_nn0 = yaml_config_handler(default_train_ae, settings.get_path2config, f'{yaml_name_index}_TrainAE')
    config_train_ae = yaml_nn0.get_class(Config_PyTorch)
    del default_train_ae, yaml_nn0

    # --- Loading the YAML file: Classifier Model training
    default_train_cl = deepcopy(DefaultSettingsTrainCE)
    default_train_cl.model_name = model_cl_default_name
    yaml_nn1 = yaml_config_handler(default_train_cl, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train_cl = yaml_nn1.get_class(Config_PyTorch)
    del default_train_cl, yaml_nn1

    # --- Processing Step #1.1: Loading dataset and Build Model
    dataset = get_dataset_ae(settings=config_data, do_classification=False,
                             mode_train_ae=settings.autoencoder_mode,
                             noise_std=settings.autoencoder_noise_std)
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
    dataset = get_dataset_cl(settings=config_data, path2model=path2folder, add_noise_cluster=add_noise_cluster)
    num_feat = dataset[0]['in'].shape[0] if not settings.autoencoder_feat_size else settings.autoencoder_feat_size
    used_model_cl = config_train_cl.get_model(input_size=num_feat, output_size=dataset.get_cluster_num)

    # --- Processing Step #1.2: Train Classifier
    metrics_cl, valid_data_cl, _ = do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train_cl,
        used_dataset=dataset, used_model=used_model_cl, path2save=path2folder
    )
    del dataset

    # --- Generate output dictionary with results
    return {"path2model_ae": path2folder, 'metric_ae': metrics_ae, 'metrics_cl': metrics_cl}


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    result = do_train_ae_classifier(DefaultSettings_MLPipe)
    print(result)
