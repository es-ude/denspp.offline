import matplotlib.pyplot as plt
import numpy as np
from os.path import join, exists
from shutil import copy
from scipy.io import loadmat
from package.plotting.plot_dnn import results_training, plot_statistic_data
from package.plotting.plot_metric import plot_confusion, plot_loss

from package.dnn.pytorch_control import Config_PyTorch, Config_Dataset
from package.dnn.pytorch_autoencoder import train_nn_autoencoder
from package.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.pytorch_classification import train_nn_classification
from package.dnn.dataset.classification import prepare_training as get_dataset_class
from package.dnn.dataset.spike_detection import prepare_training as get_dataset_sda


def __dnn_train_autoencoder(config_train: Config_PyTorch, config_data: Config_Dataset,
                            mode: int, noise_std=0.05,
                            mode_cell_bib=0, path2model="") -> None:
    """Training routine for Autoencoders
    Args:
        config_train: Settings for PyTorch Training
        config_data: Settings for Dataset Generation
        mode_ae: Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std: Std of the additional noise added to the input [default: 0.05]
        mode_cell_bib: If the dataset contains a cell library then the mode can be choicen (0: Deactivated, 1: All, 2-...: Reduced) [default: 0]
        path2model: Path to an already trained model for plotting results [default: ""]
    """
    use_cell_bib = not mode_cell_bib == 0

    only_plot_results = (path2model != "")

    if not only_plot_results:
        print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
        # --- Processing: Loading dataset and Do Training
        dataset = get_dataset_ae(settings=config_data,
                                 use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                 mode_train_ae=mode, do_classification=False,
                                 noise_std=noise_std)
        data_mean = dataset.frames_me
        trainhandler = train_nn_autoencoder(config_train, config_data)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset

        loss_ae, snr_train = trainhandler.do_training()[-1]
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()
    else:
        print(f"Plot results of already trained model")
        snr_train = list()
        logsdir = path2model
        data_result = loadmat(join(logsdir, 'results.mat'))
        data_mean = np.zeros(shape=(52, 32))

    plt.close("all")
    results_training(
        path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
        yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
        yclus=data_result['valid_clus'], snr=snr_train
    )
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])
    plt.show(block=True)


def __dnn_train_classification(config_train: Config_PyTorch, config_data: Config_Dataset,
                               mode_cell_bib=0, path2model="") -> None:
    """Training routine for Classification
        Args:
            mode_cell_bib: If the dataset contains a cell library then the mode can be choicen (0: Deactivated, 1: All, 2-...: Reduced) [default: 0]
            path2model: Path to an already trained model for plotting results [default: ""]
    """

    only_plot_results = (path2model != "")
    if not only_plot_results:
        print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
        # ---Loading Data, Do Training and getting the results
        dataset = get_dataset_class(settings=config_data,
                                    use_cell_bib=True, mode_classes=mode_cell_bib)
        trainhandler = train_nn_classification(config_train, config_data)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset, config_train

        epoch_acc = trainhandler.do_training()[-1]
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training(2)
    else:
        print(f"Plot results of already trained model")
        epoch_acc = list()
        logsdir = path2model
        data_result = loadmat(join(logsdir, 'results.mat'))

    plt.close('all')
    # plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir, epoch_zoom=[500, ])
    plot_confusion(data_result['valid_clus'], data_result['yclus'],
                   path2save=logsdir, cl_dict=data_result['cl_dict'])
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])

    plt.show(block=False)


def __dnn_train_spike_detection(config_train: Config_PyTorch, config_data: Config_Dataset,
                                path2model="") -> None:
    """Training routine for Spike Detection
    Args:
        path2model: Path to an already trained model for plotting results [default: ""]
    """
    only_plot_results = (path2model != "")

    if not only_plot_results:
        print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
        # --- Processing: Loading Data and Do Training
        dataset = get_dataset_sda(settings=config_data, threshold=4)
        dataset_dict = dataset.sda_dict
        trainhandler = train_nn_classification(config_train, config_data)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset
        epoch_acc = trainhandler.do_training()[-1]

        # --- Post-Processing: Getting data, save and plot results
        data_result = trainhandler.do_validation_after_training(3)
        logsdir = trainhandler.get_saving_path()
    else:
        print(f"Plot results of already trained model")
        epoch_acc = list()
        logsdir = path2model
        data_result = loadmat(join(logsdir, 'results.mat'))

    plt.close("all")
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=dataset_dict)
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir, cl_dict=dataset_dict)
    plt.show(block=False)


def __dnn_train_decoder(config_train: Config_PyTorch, config_data: Config_Dataset,
                        path2model="") -> None:
    """Training routine for Spike Detection
        Args:
            path2model: Path to an already trained model for plotting results [default: ""]
    """
    #TODO: Pipeline aufbauen


def check_settings_file() -> None:
    if not exists("settings_ai.py"):
        copy("package/dnn/dnn_settings_template.py", "settings_ai.py")
        print("A template configuration file is copied into main folder. Please check the content and restart!")


def do_train_dnn(mode_train: int, noise_std_ae=0.05, mode_cell_bib=0, path2trained_model="") -> None:
    """Do the Training of Deep Neural Network Topoliges
    Args:
        mode_train: 0: Autoencoder, 1: Denoising Autoencoder (Using mean), 2: Denoising Autoencoder (Adding noise),
                    3: AE+Classifier, 4: Classification, 5: Spike Detection
        noise_std_ae: Std of gaussian noise distribution [Default: 0.05]
        mode_cell_bib: Mode for using a cell bibliography [Default: 0]
        only_plot_results: Mode for only plotting the results of already trained model
    """
    from settings_ai import config_train_ae, config_train_class, config_data
    match mode_train:
        case 0:
            __dnn_train_autoencoder(config_train_ae, config_data, mode=0, noise_std=noise_std_ae,
                                    mode_cell_bib=mode_cell_bib, path2model=path2trained_model)
        case 1:
            __dnn_train_autoencoder(config_train_ae, config_data, mode=1, noise_std=noise_std_ae,
                                    mode_cell_bib=mode_cell_bib, path2model=path2trained_model)
        case 2:
            __dnn_train_autoencoder(config_train_ae, config_data, mode=2, noise_std=noise_std_ae,
                                    mode_cell_bib=mode_cell_bib, path2model=path2trained_model)
        case 4:
            __dnn_train_classification(config_train_class, config_data, mode_cell_bib=mode_cell_bib,
                                       path2model=path2trained_model)
        case 5:
            __dnn_train_spike_detection(config_train_class, config_data,
                                        path2model=path2trained_model)
        case 6:
            __dnn_train_decoder(config_train_class, config_data,
                                path2model=path2trained_model)
        case other:
            print("Wrong model!")
