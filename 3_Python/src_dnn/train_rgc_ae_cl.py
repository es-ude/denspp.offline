from copy import deepcopy
from package.yaml_handler import yaml_config_handler
from package.plot.plot_dnn import results_training
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import (Config_Dataset, DefaultSettingsDataset,
                                           Config_PyTorch, DefaultSettingsTrainCE, DefaultSettingsTrainMSE)
from package.dnn.pytorch_pipeline import do_train_autoencoder, do_train_classifier, get_model_attributes
from package.data_process.rgc_combination import rgc_logic_combination

from package.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.dataset.autoencoder_class import prepare_training as get_dataset_cl
import src_dnn.models.rgc_ae_cl as models


def do_train_rgc_ae_cl(settings: Config_ML_Pipeline, yaml_name_index='Config_RGC', add_noise_cluster=False) -> dict:
    """Training routine for Autoencoders and Classification after Encoder for Retinal Ganglion Celltype Classification
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:    Index of yaml file name
        add_noise_cluster:  Adding noise cluster to dataset [Default: False]
    Returns:
        Dictionary with metric results from Autoencoder and Classifier Training
    """

    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = '2023-11-24_Dataset-07_RGC_TDB_Merged.npy'
    yaml_data = yaml_config_handler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Autoencoder Training
    default_ae = deepcopy(DefaultSettingsTrainMSE)
    default_ae.model_name = get_model_attributes(models, '_ae_v')
    yaml_train_ae = yaml_config_handler(default_ae, settings.get_path2config, f'{yaml_name_index}_TrainAE')
    config_train_ae = yaml_train_ae.get_class(Config_PyTorch)

    # --- Loading the YAML file: Classifier Training
    default_cl = deepcopy(DefaultSettingsTrainCE)
    default_cl.model_name = get_model_attributes(models, '_cl_v')
    yaml_train_cl = yaml_config_handler(default_cl, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train_cl = yaml_train_cl.get_class(Config_PyTorch)

    del default_data, default_ae, default_cl
    del yaml_data, yaml_train_ae, yaml_train_cl

    # --- Processing Step #1.1: Loading dataset and Build Model
    print("\n# ----------- Step #1: TRAINING AUTOENCODER")
    dataset = get_dataset_ae(settings=config_data, do_classification=False,
                             mode_train_ae=settings.autoencoder_mode,
                             noise_std=settings.autoencoder_noise_std,
                             add_noise_cluster=add_noise_cluster)
    if settings.autoencoder_feat_size:
        used_model_ae = models.models_available.build_model(config_train_ae.model_name,
                                                            output_size=settings.autoencoder_feat_size)
    else:
        used_model_ae = models.models_available.build_model(config_train_ae.model_name)

    metrics_ae, valid_data_ae, path2folder = do_train_autoencoder(
        config_ml=settings, config_data=config_data, config_train=config_train_ae,
        used_dataset=dataset, used_model=used_model_ae, path2save='', calc_custom_metrics=['snr']
    )
    if settings.do_plot:
        used_first_fold = [key for key in metrics_ae.keys()][0]
        results_training(
            path=path2folder, cl_dict=dataset.get_dictionary, feat=valid_data_ae['feat'],
            yin=valid_data_ae['input'], ypred=valid_data_ae['pred'], ymean=dataset.get_mean_waveforms,
            yclus=valid_data_ae['valid_clus'], snr=metrics_ae[used_first_fold]['snr']
        )
    del dataset

    # --- Processing: Loading dataset and Do Classification
    print("\n# ----------- Step #2: TRAINING CLASSIFIER")
    dataset = get_dataset_cl(settings=config_data, path2model=path2folder, add_noise_cluster=add_noise_cluster)
    num_feat = dataset[0]['in'].shape[0] if not settings.autoencoder_feat_size else settings.autoencoder_feat_size
    used_model_cl = models.models_available.build_model(config_train_cl.model_name, input_size=num_feat)

    metrics_cl, valid_data_cl, _ = do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train_cl,
        used_dataset=dataset, used_model=used_model_cl, path2save=path2folder
    )
    if settings.do_plot:
        rgc_logic_combination(path2folder, show_plot=settings.do_block)
    del dataset

    # --- Generate output dictionary with results
    return {"path2model_ae": path2folder, 'metric_ae': metrics_ae, 'metrics_cl': metrics_cl}


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_rgc_ae_cl(DefaultSettings_MLPipe)
