from torch import nn
import matplotlib.pyplot as plt

from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import package.dnn.template.models.autoencoder_cnn as models


config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.cnn_ae_v4(32, 12),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    patience=20,
    num_epochs=10,
    batch_size=256,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def do_train_ae(dnn_handler: dnn_handler, mode_ae: int, noise_std=0.05) -> None:
    """Training routine for Autoencoders
    Args:
        dnn_handler: Handler for configuring the routine selection for train deep neural networks
        mode_ae: Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std: Std of the additional noise added to the input [default: 0.05]
    """
    from package.dnn.template.dataset.autoencoder import prepare_training
    from package.dnn.pytorch.autoencoder import train_nn
    from package.plot.plot_dnn import results_training, plot_statistic_data

    # --- Loading the YAML files
    yaml_data = yaml_config_handler(models.Recommended_Config_DatasetSettings, yaml_name='Config_AE_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    use_cell_bib = not (dnn_handler.mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else dnn_handler.mode_cell_bib - 1
    # --- Processing: Loading dataset and Do Training
    dataset = prepare_training(settings=config_data,
                               use_cell_bib=use_cell_bib, mode_classes=use_cell_mode,
                               mode_train_ae=mode_ae, do_classification=False,
                               noise_std=noise_std)
    data_mean = dataset.frames_me
    trainhandler = train_nn(config_train, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    loss_ae, snr_train = trainhandler.do_training(metrics='snr')[-1]

    # --- Post-Processing: Validation after training
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()

    # --- Plotting and Ending
    if dnn_handler.do_plot:
        plt.close("all")
        results_training(
            path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
            yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
            yclus=data_result['valid_clus'], snr=snr_train
        )
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'], show_plot=dnn_handler.do_block)
