import matplotlib.pyplot as plt
from package.yaml_handler import yaml_config_handler

from package.dnn.pytorch.autoencoder import train_nn as train_autoencoder
from package.dnn.pytorch.classifier import train_nn as train_classifier
from package.plot.plot_dnn import results_training, plot_statistic_data
from package.plot.plot_metric import plot_confusion, plot_loss
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_dataclass import (Config_Dataset, DefaultSettingsDataset,
                                           Config_PyTorch, DefaultSettingsTrainCE, DefaultSettingsTrainMSE)

from package.dnn.template.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.template.dataset.autoencoder_class import prepare_training as get_dataset_class
import src_dnn.models.rgc_ae_cl as models
from src_dnn.train_rgc_class import rgc_logic_combination


def do_train_rgc_ae_cl(settings: dnn_handler,
                       num_feature_layer=0,
                       mode_ae=0, noise_std=0.05) -> None:
    """Training routine for Autoencoders and Classification after Encoder
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        num_feature_layer:  Size of feature layer after decoding of autoencoder [Default: 0 = take default]
        mode_ae:            Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std:          Std of the additional noise added to the input [default: 0.05]
    Return:
        None
    """
    # --- Loading the YAML files
    default_data = DefaultSettingsDataset
    default_data.data_path = 'data'
    default_data.data_file_name = '2023-11-24_Dataset-07_RGC_TDB_Merged.mat'
    yaml_data = yaml_config_handler(default_data, yaml_name='Config_RGC_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del default_data, yaml_data

    default_ae = DefaultSettingsTrainMSE
    default_ae.model_name = models.cnn_rgc_ae_v1.__name__
    yaml_train_ae = yaml_config_handler(default_ae)
    config_train_ae = yaml_train_ae.get_class(Config_PyTorch)
    del default_ae, yaml_train_ae

    default_cl = DefaultSettingsTrainCE
    default_cl.model_name = models.rgc_ae_cl_v1.__name__
    yaml_train_cl = yaml_config_handler(default_cl)
    config_train_cl = yaml_train_cl.get_class(Config_PyTorch)
    del default_cl, yaml_train_cl

    metric_snr_run = list()
    # ----------- Step #1: TRAINING AUTOENCODER
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = get_dataset_ae(settings=config_data, mode_train_ae=mode_ae, noise_std=noise_std, do_classification=False)

    trainhandler = train_autoencoder(config_train=config_train_ae, config_data=config_data)

    if num_feature_layer:
        used_model_ae = models.models_available.build_model(config_train_ae.model_name, output_size=num_feature_layer)
    else:
        used_model_ae = models.models_available.build_model(config_train_ae.model_name)
    trainhandler.load_model(used_model_ae)
    trainhandler.load_data(dataset)
    loss_ae, snr_ae = trainhandler.do_training(metrics='snr')[-1]
    path2model = trainhandler.get_saving_path()

    if settings.do_plot:
        plt.close('all')
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()
        data_mean = dataset.frames_me

        plot_loss(loss_ae, 'Loss', path2save=logsdir)
        results_training(
            path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
            yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
            yclus=data_result['valid_clus'], snr=snr_ae
        )
    del dataset, trainhandler

    # ----------- Step #2: TRAINING CLASSIFIER
    # --- Processing: Loading dataset and Do Classification
    dataset = get_dataset_class(settings=config_data, path2model=path2model)
    trainhandler = train_classifier(config_train=config_train_cl, config_data=config_data)

    num_feat = dataset[0]['in'].shape[0] if not num_feature_layer else num_feature_layer
    used_model_cl = models.models_available.build_model(config_train_cl.model_name,
                                                        input_size=num_feat)
    trainhandler.load_model(used_model_cl)
    trainhandler.load_data(dataset)
    loss_class, acc_class = trainhandler.do_training()[-1]

    if settings.do_plot:
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()

        plot_loss(loss_class, 'Loss', path2save=logsdir)
        plot_loss(acc_class, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       cl_dict=data_result['cl_dict'], path2save=logsdir,
                       name_addon="training")
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        rgc_logic_combination(logsdir, show_plot=settings.do_block)

    del dataset, trainhandler
    print("\nThe End")
