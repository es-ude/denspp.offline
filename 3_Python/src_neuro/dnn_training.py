import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from shutil import copy


def __dnn_train_ae(mode_ae: int, noise_std=0.05, mode_cell_bib=0, do_plot=True, block_plot=True) -> None:
    """Training routine for Autoencoders
    Args:
        mode_ae: Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std: Std of the additional noise added to the input [default: 0.05]
        mode_cell_bib: If dataset has a cell library then mode decides (0: Deactivated, 1: All, 2: Reduced) [default:0]
        do_plot: Doing the plots during the training routine
        block_plot: Blocking the plot outputs if do_plot is active
    """
    from settings_ai import config_data, config_train_ae
    from package.dnn.dataset.autoencoder import prepare_training
    from package.dnn.pytorch.autoencoder import train_nn
    from package.plot.plot_dnn import results_training, plot_statistic_data

    use_cell_bib = not (mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else mode_cell_bib - 1

    # --- Processing: Loading dataset and Do Training
    dataset = prepare_training(settings=config_data,
                               use_cell_bib=use_cell_bib, mode_classes=use_cell_mode,
                               mode_train_ae=mode_ae, do_classification=False,
                               noise_std=noise_std)
    data_mean = dataset.frames_me
    trainhandler = train_nn(config_train_ae, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset

    loss_ae, snr_train = trainhandler.do_training()[-1]
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if do_plot:
        plt.close("all")
        results_training(
            path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
            yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
            yclus=data_result['valid_clus'], snr=snr_train
        )
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        plt.show(block=block_plot)


def __dnn_train_ae_class(mode_ae: int, noise_std=0.05, mode_cell_bib=0, do_plot=False, block_plot=False) -> None:
    """Training routine for Autoencoders
    Args:
        mode_ae: Mode of Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std: Std of the additional noise added to the input [default: 0.05]
        mode_cell_bib: If dataset has a cell library then mode decides (0: Deactivated, 1: All, 2: Reduced) [default:0]
        do_plot: Doing the plots during the training routine
        block_plot: Blocking the plot outputs if do_plot is active
    """
    from settings_ai import config_data
    from settings_ai import config_train_ae, config_train_ae_class
    from package.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
    from package.dnn.dataset.autoencoder_class import prepare_training as get_dataset_class
    from package.dnn.pytorch.autoencoder import train_nn as train_autoencoder
    from package.dnn.pytorch.classifier import train_nn as train_classifier
    from package.plot.plot_dnn import results_training, plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    use_cell_bib = not (mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else mode_cell_bib - 1

    metric_snr_run = list()
    # ----------- Step #1: TRAINING AUTOENCODER
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = get_dataset_ae(settings=config_data, use_cell_bib=use_cell_bib, mode_classes=use_cell_mode,
                             mode_train_ae=mode_ae, noise_std=noise_std, do_classification=False)
    trainhandler = train_autoencoder(config_train=config_train_ae, config_data=config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    loss_ae, snr_ae = trainhandler.do_training()[-1]
    path2model = trainhandler.get_saving_path()

    if do_plot:
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
                                use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib)
    trainhandler = train_classifier(config_train=config_train_ae_class, config_data=config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    acc_class = trainhandler.do_training()[-1]

    if do_plot:
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()

        plot_loss(acc_class, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       cl_dict=data_result['cl_dict'], path2save=logsdir,
                       name_addon="training")
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        plt.show(block=block_plot)
    else:
        # --- Übergabe next run (Taking best results
        last_loss = loss_ae[-1]
        last_snr = snr_ae[-1].detach().numpy()
        last_snr = (last_snr.min(), np.median(last_snr), last_snr.max())
        last_class = acc_class[-1]

        metric_snr_run.append((last_loss, last_snr, last_class))

    del dataset, trainhandler


def __dnn_train_rgc_class(mode_cell_bib=0, do_plot=True, block_plot=True) -> None:
    """Training routine for Classification
    Args:
        mode_cell_bib: If dataset has a cell library then mode decides (0: Deactivated, 1: All, 2: Reduced) [default:0]
        do_plot: Doing the plots during the training routine
        block_plot: Blocking the plot outputs if do_plot is active
    """
    from settings_ai import config_data, config_train_rgc
    from package.dnn.dataset.rgc_classification import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    use_cell_bib = not (mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else mode_cell_bib-1

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(settings=config_data, use_cell_bib=use_cell_bib, mode_classes=use_cell_mode)
    trainhandler = train_nn(config_train_rgc, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training(2)
    del trainhandler

    # --- Plotting
    if do_plot:
        plt.close('all')
        # plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir, epoch_zoom=[500, ])
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       path2save=logsdir, cl_dict=data_result['cl_dict'])
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])

        plt.show(block=block_plot)


def __dnn_train_sda(do_plot=True, block_plot=True) -> None:
    """Training routine for Spike Detection
    Args:
        do_plot: Doing the plots during the training routine
        block_plot: Blocking the plot outputs if do_plot is active
    """
    from settings_ai import config_data, config_train_sda
    from package.dnn.dataset.spike_detection import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(settings=config_data, threshold=4)
    data_dict = dataset.sda_dict
    trainhandler = train_nn(config_train_sda, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training(3)
    logsdir = trainhandler.get_saving_path()
    del trainhandler

    # --- Plotting
    if do_plot:
        plt.close("all")
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=data_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir, cl_dict=data_dict)
        plt.show(block=block_plot)


def __dnn_train_decoder(do_plot=True, block_plot=True) -> None:
    """Training routine for Neural Decoding
   Args:
       do_plot: Doing the plots during the training routine
       block_plot: Blocking the plot outputs if do_plot is active
    """
    from settings_ai import config_data, config_train_dec
    from package.dnn.dataset.decoding_utah import prepare_training
    from package.dnn.pytorch.rnn import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(config_data, 500)
    data_dict = dataset.frame_dict
    trainhandler = train_nn(config_train_dec, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training(3)
    logsdir = trainhandler.get_saving_path()
    del trainhandler

    # --- Plotting
    if do_plot:
        plt.close("all")
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=data_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir, cl_dict=data_dict)
        plt.show(block=block_plot)


def check_settings_file() -> None:
    if not exists("settings_ai.py"):
        copy("package/dnn/dnn_settings_template.py", "settings_ai.py")
        print("A template configuration file is copied into main folder. Please check the content and restart!")


def do_train_dnn(mode_train: int, noise_std_ae=0.05, mode_cell_bib=0, do_plot=True, block_plot=True) -> None:
    """Do the Training of Deep Neural Networks
    Args:
        mode_train: 0: Autoencoder, 1: Denoising Autoencoder (Using mean), 2: Denoising Autoencoder (Adding noise),
                    3: AE+Classifier, 4: Classification, 5: Spike Detection, 6: Neural Decoder
        noise_std_ae: Std of gaussian noise distribution [Default: 0.05]
        mode_cell_bib: If dataset has a cell library then mode decides (0: Deactivated, 1: All, 2: Reduced) [default:0]
        do_plot: Doing the plots during the training routine
        block_plot: Blocking the plot outputs if do_plot is active
    """
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)")
    match mode_train:
        case 0:
            __dnn_train_ae(mode_ae=0, noise_std=noise_std_ae, mode_cell_bib=mode_cell_bib,
                           do_plot=do_plot, block_plot=block_plot)
        case 1:
            __dnn_train_ae(mode_ae=1, noise_std=noise_std_ae, mode_cell_bib=mode_cell_bib,
                           do_plot=do_plot, block_plot=block_plot)
        case 2:
            __dnn_train_ae(mode_ae=2, noise_std=noise_std_ae, mode_cell_bib=mode_cell_bib,
                           do_plot=do_plot, block_plot=block_plot)
        case 3:
            __dnn_train_ae_class(mode_ae=0, noise_std=noise_std_ae, mode_cell_bib=mode_cell_bib,
                                 do_plot=do_plot, block_plot=block_plot)
        case 4:
            __dnn_train_rgc_class(mode_cell_bib=mode_cell_bib, do_plot=do_plot, block_plot=block_plot)
        case 5:
            __dnn_train_sda(do_plot=do_plot, block_plot=block_plot)
        case 6:
            __dnn_train_decoder(do_plot=do_plot, block_plot=block_plot)
        case _:
            print("Wrong model!")

    print("\nThe End")
