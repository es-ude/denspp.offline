import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import package.dnn.models.autoencoder_cnn as models_ae
import package.dnn.models.autoencoder_class as models_class


config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    #data_file_name='2023-06-30_Dataset03_SimDaten_Quiroga2020_Sorted',
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=True,
    data_normalization_mode='CPU',
    data_normalization_method='minmax',
    data_normalization_setting='bipolar',
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)


def do_train_ae_classifier(dnn_handler: dnn_handler,
                           num_feature_layer: int, num_output_cl: int,
                           mode_ae=0, noise_std=0.05) -> dict:
    """Training routine for Autoencoders and Classification after Encoder
    Args:
        dnn_handler:        Handler for configurating the routine selection for train deep neural networks
        num_feature_layer:  Size of hidden layer from autoencoder
        num_output_cl:      Output size of classifier
        mode_ae:            Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std:          Std of the additional noise added to the input [default: 0.05]
    """
    from package.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
    from package.dnn.dataset.autoencoder_class import prepare_training as get_dataset_class
    from package.dnn.pytorch.autoencoder import train_nn as train_autoencoder
    from package.dnn.pytorch.classifier import train_nn as train_classifier
    from package.plot.plot_dnn import results_training, plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    metric_run = dict()

    # --- Definition of settings
    config_train_ae = Config_PyTorch(
        # --- Settings of Models/Training
        model=models_ae.cnn_ae_v4(32, num_feature_layer),
        loss='MSE',
        loss_fn=nn.MSELoss(),
        optimizer='Adam',
        num_kfold=1,
        patience=20,
        num_epochs=100,
        batch_size=256,
        data_split_ratio=0.25,
        data_do_shuffle=True
    )
    config_train_cl = Config_PyTorch(
        # --- Settings of Models/Training
        model=models_class.classifier_ae_v1(num_feature_layer, num_output_cl),
        loss='Cross Entropy',
        loss_fn=nn.CrossEntropyLoss(),
        optimizer='Adam',
        num_kfold=1,
        patience=20,
        num_epochs=100,
        batch_size=256,
        data_split_ratio=0.25,
        data_do_shuffle=True
    )

    use_cell_bib = not (dnn_handler.mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else dnn_handler.mode_cell_bib - 1

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
    # --- Processing: Loading dataset and Do Classification
    dataset = get_dataset_class(settings=config_data, path2model=path2model,
                                use_cell_bib=use_cell_bib, mode_classes=use_cell_mode)
    num_output = dataset.frames_me.shape[0]
    trainhandler = train_classifier(config_train=config_train_cl, config_data=config_data)
    trainhandler.load_model()
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
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        plt.show(block=dnn_handler.do_block)

    del dataset, trainhandler

    # --- Generierung output
    metric_run.update({"path2save": logsdir})
    return metric_run


if __name__ == "__main__":
    from package.dnn.dnn_handler import dnn_handler
    import matplotlib.pyplot as plt

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

    # --- Plotten
    plt.figure()
    plt.show()
