from torch import nn
import matplotlib.pyplot as plt

from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import src_dnn.models.rgc_ae_cl as models
from src_dnn.train_rgc_class import rgc_logic_combination


def do_train_rgc_ae_cl(dnn_handler: dnn_handler,
                       num_feature_layer: int, num_output: int,
                       mode_ae: int, noise_std=0.05) -> None:
    """Training routine for Autoencoders and Classification after Encoder
    Args:
        dnn_handler: Handler for configurating the routine selection for train deep neural networks
        mode_ae: Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std: Std of the additional noise added to the input [default: 0.05]
    """
    from package.dnn.template.dataset.autoencoder import prepare_training as get_dataset_ae
    from package.dnn.template.dataset.autoencoder_class import prepare_training as get_dataset_class
    from package.dnn.pytorch.autoencoder import train_nn as train_autoencoder
    from package.dnn.pytorch.classifier import train_nn as train_classifier
    from package.plot.plot_dnn import results_training, plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    # --- Definition of settings
    config_train_ae = Config_PyTorch(
        # --- Settings of Models/Training
        model=models.cnn_rgc_ae_v1(32, num_feature_layer),
        loss='MSE',
        loss_fn=nn.MSELoss(),
        optimizer='Adam',
        num_kfold=1,
        num_epochs=100,
        batch_size=1024,
        data_split_ratio=0.25,
        data_do_shuffle=True
    )
    config_train_cl = Config_PyTorch(
        # --- Settings of Models/Training
        model=models.rgc_ae_cl_v2(num_feature_layer, num_output),
        loss='Cross Entropy',
        loss_fn=nn.CrossEntropyLoss(),
        optimizer='Adam',
        num_kfold=1,
        num_epochs=100,
        batch_size=1024,
        data_split_ratio=0.25,
        data_do_shuffle=True
    )

    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)")
    use_cell_bib = not (dnn_handler.mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else dnn_handler.mode_cell_bib - 1

    # --- Loading the YAML files
    yaml_data = yaml_config_handler(models.RecommendedRGCDataset, yaml_name='Config_RGC_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    metric_snr_run = list()
    # ----------- Step #1: TRAINING AUTOENCODER
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = get_dataset_ae(settings=config_data, use_cell_bib=use_cell_bib, mode_classes=use_cell_mode,
                             mode_train_ae=mode_ae, noise_std=noise_std, do_classification=False)
    trainhandler = train_autoencoder(config_train=config_train_ae, config_data=config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    loss_ae, snr_ae = trainhandler.do_training(metrics='snr')[-1]
    path2model = trainhandler.get_saving_path()

    if dnn_handler.do_plot:
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
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])

    del dataset, trainhandler

    # ----------- Step #2: TRAINING CLASSIFIER
    # --- Processing: Loading dataset and Do Classification
    dataset = get_dataset_class(settings=config_data, path2model=path2model,
                                use_cell_bib=use_cell_bib, mode_classes=use_cell_mode)
    num_output = dataset.frames_me.shape[0]
    trainhandler = train_classifier(config_train=config_train_cl, config_data=config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    loss_class, acc_class = trainhandler.do_training()[-1]

    if dnn_handler.do_plot:
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training(num_output)

        plot_loss(loss_class, 'Loss', path2save=logsdir)
        plot_loss(acc_class, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       cl_dict=data_result['cl_dict'], path2save=logsdir,
                       name_addon="training")
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        rgc_logic_combination(logsdir, show_plot=dnn_handler.do_block)

    del dataset, trainhandler
    print("\nThe End")
