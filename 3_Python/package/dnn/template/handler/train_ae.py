import matplotlib.pyplot as plt
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsTrainMSE,
                                           Config_Dataset, DefaultSettingsDataset)
from package.plot.plot_dnn import results_training, plot_statistic_data

from package.dnn.template.dataset.autoencoder import prepare_training
from package.dnn.pytorch.autoencoder import train_nn
import package.dnn.template.models.autoencoder_dnn as models


def do_train_ae(settings: dnn_handler, mode_ae=0, noise_std=0.05, num_feat_layer=0, add_noise_cluster=False) -> None:
    """Training routine for Autoencoders in Neural Applications (Spike Frames)
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        mode_ae:            Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input))
        num_feat_layer:     Size of feature layer [Default: 0 = take default]
        noise_std:          Std of the additional noise added to the input
        add_noise_cluster:  Decision for adding noise cluster activity
    Returns:
        None
    """
    # --- Loading the YAML files
    yaml_data = yaml_config_handler(DefaultSettingsDataset, yaml_name='Config_AE_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del yaml_data

    default_train = DefaultSettingsTrainMSE
    default_train.model_name = models.dnn_ae_v1.__name__
    yaml_nn = yaml_config_handler(default_train, yaml_name='Config_AE_Training')
    config_train = yaml_nn.get_class(Config_PyTorch)
    del default_train, yaml_nn

    # --- Processing: Loading dataset and Do Training
    dataset = prepare_training(settings=config_data, do_classification=False,
                               mode_train_ae=mode_ae, noise_std=noise_std,
                               add_noise_cluster=add_noise_cluster,
                               use_median_for_mean=True)
    data_mean = dataset.frames_me

    trainhandler = train_nn(config_train, config_data)
    if num_feat_layer:
        model = models.models_available.build_model(config_train.model_name, output_size=num_feat_layer)
    else:
        model = models.models_available.build_model(config_train.model_name)
    trainhandler.load_model(model)
    trainhandler.load_data(dataset)
    del dataset
    loss_ae, snr_train = trainhandler.do_training(metrics='snr')[-1]

    # --- Post-Processing: Validation after training
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()

    # --- Plotting and Ending
    if settings.do_plot:
        plt.close('all')
        results_training(
            path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
            yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
            yclus=data_result['valid_clus'], snr=snr_train
        )
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'],
                            show_plot=settings.do_block)
    print("\nThe End")


if __name__ == "__main__":
    set0 = dnn_handler(
        mode_train_dnn=0,
        do_plot=True,
        do_block=True
    )
    do_train_ae(set0)
