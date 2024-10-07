import numpy as np
import matplotlib.pyplot as plt

from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE,
                                           Config_Dataset, DefaultSettingsDataset)

import package.dnn.template.models.autoencoder_cnn as models_ae
import package.dnn.template.models.autoencoder_class as models_class


def do_train_ae_classifier(dnn_handler: dnn_handler,
                           num_feat_layer: int, num_output_cl: int, add_noise_cluster=False,
                           mode_ae=0, noise_std=0.05) -> dict:
    """Training routine for Autoencoders and Classification after Encoder
    Args:
        dnn_handler:        Handler for configurating the routine selection for train deep neural networks
        num_feat_layer:     Size of hidden layer from autoencoder
        num_output_cl:      Output size of classifier
        add_noise_cluster:  Adding noise cluster to dataset [Default: False]
        mode_ae:            Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input))
        noise_std:          Std of the additional noise added to the input [default: 0.05]
    Returns:
        Dictionary
    """
    from package.dnn.template.dataset.autoencoder import prepare_training as get_dataset_ae
    from package.dnn.template.dataset.autoencoder_class import prepare_training as get_dataset_cl
    from package.dnn.pytorch.autoencoder import train_nn as train_autoencoder
    from package.dnn.pytorch.classifier import train_nn as train_classifier
    from package.plot.plot_dnn import results_training, plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    metric_run = dict()
    # --- Loading the YAML files
    yaml_data = yaml_config_handler(DefaultSettingsDataset, yaml_name='Config_AE+CL_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    yaml_nn0 = yaml_config_handler(DefaultSettingsTrainMSE, yaml_name='Config_AECL_TrainAE')
    config_train_ae = yaml_nn0.get_class(Config_PyTorch)
    model_ae = models_ae.models_available.build_model(config_train_ae.model_name, output_size=num_feat_layer)

    # ----------- Step #1: TRAINING AUTOENCODER
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = get_dataset_ae(settings=config_data, do_classification=False,
                             mode_train_ae=mode_ae, noise_std=noise_std, add_noise_cluster=add_noise_cluster)
    trainhandler = train_autoencoder(config_train=config_train_ae, config_data=config_data)
    trainhandler.load_model(model_ae)
    trainhandler.load_data(dataset)
    loss_ae, snr_ae = trainhandler.do_training(metrics='snr')[-1]
    path2model = trainhandler.get_saving_path()

    if dnn_handler.do_plot:
        plt.close('all')
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()
        data_mean = dataset.frames_me

        results_training(
            path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
            yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
            yclus=data_result['valid_clus'], snr=snr_ae
        )
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])

    del dataset, trainhandler

    # ----------- Step #2: TRAINING CLASSIFIER
    yaml_nn1 = yaml_config_handler(DefaultSettingsTrainCE, yaml_name='Config_AECL_TrainCL')
    config_train_cl = yaml_nn1.get_class(Config_PyTorch)
    model_cl = models_class.models_available.build_model(config_train_cl.model_name,
                                                         input_size=num_feat_layer, output_size=num_output_cl)

    # --- Processing: Loading dataset and Do Classification
    dataset = get_dataset_cl(settings=config_data, path2model=path2model, add_noise_cluster=add_noise_cluster)
    num_output = dataset.frames_me.shape[0]
    trainhandler = train_classifier(config_train=config_train_cl, config_data=config_data)
    trainhandler.load_model(model_cl)
    trainhandler.load_data(dataset)
    acc_class = trainhandler.do_training()[-1]

    if dnn_handler.do_plot:
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training(num_output)

        plot_loss(acc_class, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       cl_dict=data_result['cl_dict'], path2save=logsdir,
                       name_addon="training")
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'],
                            show_plot=dnn_handler.do_block)

    del dataset, trainhandler

    # --- Generierung output
    metric_run.update({"path2save": logsdir})
    return metric_run


if __name__ == "__main__":
    from package.dnn.dnn_handler import dnn_handler

    dnn_handler = dnn_handler(
        mode_dnn=0,
        mode_cellbib=0,
        do_plot=True,
        do_block=True
    )
    size_hidden_layer = np.arange(1, 20, 3, dtype=int).tolist()

    # --- Iteration
    metrics_runs = dict()
    for idx, hidden_size in enumerate(size_hidden_layer):
        result = do_train_ae_classifier(dnn_handler, hidden_size, 4)

        result.update({"Size_Hiddenlayer": hidden_size})
        metrics_runs.update({f"Run_{idx:03d}": result})
