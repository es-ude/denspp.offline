import matplotlib.pyplot as plt
import numpy as np
import csv, os
from os.path import join, exists
from shutil import copy
from scipy.io import loadmat
from package.plotting.plot_dnn import results_training, plot_statistic_data
from package.plotting.plot_metric import prep_confusion, plot_loss, plot_confusion

from package.dnn.pytorch_control import Config_PyTorch, Config_Dataset
from package.dnn.pytorch_autoencoder import train_nn_autoencoder
from package.dnn.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.pytorch_classification import train_nn_classification
from package.dnn.dataset.classification import prepare_training as get_dataset_class
from package.dnn.dataset.autoencoder_class import prepare_training as get_dataset_ae_class
from package.dnn.dataset.spike_detection import prepare_training as get_dataset_sda


def __dnn_train_autoencoder(config_data: Config_Dataset,
                            config_train: Config_PyTorch,
                            mode: int, noise_std=0.05,
                            mode_cell_bib=0, only_plot_results=False) -> None:
    """Training routine for Autoencoders
    Args:
        mode_ae: Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input)) [default:0]
        noise_std: Std of the additional noise added to the input [default: 0.05]
        mode_cell_bib: If the dataset contains a cell library then the mode can be choicen (0: Deactivated, 1: All, 2-...: Reduced) [default: 0]
        only_plot_results: Plotting the results of a already trained model [default: False]
    """
    use_cell_bib = not mode_cell_bib == 0

    if not only_plot_results:
        # --- Processing: Loading dataset and Do Training
        dataset = get_dataset_ae(path2data=config_data.get_path2data(), data_settings=config_data,
                                 use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                 mode_train_ae=mode, do_classification=False,
                                 noise_std=noise_std)
        data_mean = dataset.frames_me
        trainhandler = train_nn_autoencoder(config_train, config_data)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset

        loss, snr = trainhandler.do_training()[-1]
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()
    else:
        print(f"Plot results of already trained model")
        snr = list()
        logsdir = 'runs/20231201_165901_train_cnn_ae_v3/'
        data_result = loadmat(join(logsdir, 'results.mat'))
        data_mean = np.zeros(shape=(52, 32))

    plt.close("all")
    results_training(
        path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
        yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
        yclus=data_result['valid_clus'], snr=snr
    )
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])
    plt.show(block=True)


def __dnn_train_combi(config_dataset: Config_Dataset, config_train_ae: Config_PyTorch, config_train_class: Config_PyTorch,
                      noise_std: 0.00, use_cell_bib=True, mode_cell_bib=1,  do_plot=True):
    plt.close("all")
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
    print("Training Autoencoder started")

    metric_snr_run = list()
    # ----------- Step #1: TRAINING AUTOENCODER
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = get_dataset_ae(path2data=config_dataset.get_path2data(), data_settings=config_dataset,
                             use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib, noise_std=noise_std)

    trainhandler = train_nn_autoencoder(config_train=config_train_ae, config_dataset=config_dataset)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    loss_ae, snr_ae = trainhandler.do_training()[-1]
    path2model = trainhandler.get_saving_path()
    # --- Reducing
    used_loss = loss_ae[-1]
    used_snr = snr_ae[-1]
    used_snr = (used_snr.min(), np.median(used_snr), used_snr.max())

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
        plt.show(block=True)

    del dataset, trainhandler
    print("Training Autoencoder ended")

    # ----------- Step #2: TRAINING CLASSIFIER
    # --- Processing: Loading dataset and Do Classification
    dataset = get_dataset_ae_class(path2data=config_dataset.get_path2data(), data_settings=config_dataset,
                                   path2model=path2model,
                                   use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                   noise_std=noise_std)
    trainhandler = train_nn_classification(config_train=config_train_class, config_dataset=config_dataset)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    acc_class = trainhandler.do_training(path2save=path2model)[-1]
    # --- Reducing
    used_acc = acc_class[-1]

    if do_plot:
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()

        plot_loss(acc_class, 'Acc.', path2save=logsdir)
        prep_confusion(data_result['valid_clus'], data_result['yclus'], "training", "both", False,
                       cl_dict=data_result['cl_dict'], path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        plt.show(block=False)
    else:
        # --- Übergabe next run
        metric_snr_run.append((used_loss, used_snr, used_acc))

    logsdirect = trainhandler.get_saving_path()
    del dataset, trainhandler

    # Specify the folder and file name
    folder_path = logsdirect
    file_name = "Results_Loss_SNR_Acc.csv"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Specify the complete file path
    csv_file_path = os.path.join(folder_path, file_name)

    # Writing to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow("Loss")
        writer.writerow(used_loss)
        writer.writerow("SNR")
        writer.writerow(used_snr)
        writer.writerow("Accuracy")
        writer.writerow(used_acc)
    print("ENDE")


def __dnn_train_classification(config_data: Config_Dataset, config_train: Config_PyTorch,
                               mode_cell_bib=0, only_plot_results=False) -> None:
    """Training routine for Classification
        Args:
            mode_cell_bib: If the dataset contains a cell library then the mode can be choicen (0: Deactivated, 1: All, 2-...: Reduced) [default: 0]
            only_plot_results: Plotting the results of a already trained model [default: False]
    """
    if not only_plot_results:
        # ---Loading Data, Do Training and getting the results
        dataset = get_dataset_class(path=config_data.get_path2data(), settings=config_data,
                                   use_cell_bib=True, mode_classes=mode_cell_bib)
        trainhandler = train_nn_classification(config_train, config_data)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset, config_train

        epoch_acc = trainhandler.do_training()[-1]
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training(2)
    else:
        epoch_acc = list()
        logsdir = 'runs/20231204_004158_train_rgc_class_v2/'
        data_result = loadmat(join(logsdir, 'results.mat'))

    plt.close('all')
    # plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir, epoch_zoom=[500, ])
    prep_confusion(data_result['valid_clus'], data_result['yclus'],
                   path2save=logsdir, cl_dict=data_result['cl_dict'])
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])

    plt.show(block=False)


def __dnn_train_spike_detection(config_data: Config_Dataset, config_train: Config_PyTorch,
                                only_plot_results=False) -> None:
    """Training routine for Spike Detection
    Args:
        only_plot_results: Plotting the results of a already trained model [default: False]
    """
    # --- Processing: Loading Data and Do Training
    dataset = get_dataset_sda(path=config_data.get_path2data(), settings=config_data, threshold=4)
    dataset_dict = dataset.sda_dict
    trainhandler = train_nn_classification(config_train, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training(3)

    logsdir = trainhandler.get_saving_path()

    plt.close("all")
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=dataset_dict)
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir, cl_dict=dataset_dict)
    plt.show(block=False)


def check_settings_file() -> None:
    if not exists("settings_ai.py"):
        copy("package/dnn/dnn_settings_template.py", "settings_ai.py")
        print("A template configuration file is copied into main folder. Please check the content and restart!")


def do_train_dnn(mode_train: int, noise_std_ae=0.05, mode_cell_bib=0, only_plot_results=False, path2model='') -> None:
    """Do the Training of Deep Neural Network Topoliges
    Args:
        mode_train: 0: Autoencoder, 1: Denoising Autoencoder (Use mean as output), 2: Denoising Autoencoder (More noise on input)
        noise_std_ae: Std of gaussian noise distribution [Default: 0.05]
        mode_cell_bib: Mode for using a cell bibliography [Default: 0]
        only_plot_results: Mode for only plotting the results of already trained model
        path2model: Path to already trained model
    """
    from settings_ai import config_dataset, config_train_ae, config_train_class, config_train_rgc
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)")
    match mode_train:
        case 0:
            __dnn_train_autoencoder(config_dataset, config_train_ae, mode=0, noise_std=noise_std_ae,
                                    mode_cell_bib=mode_cell_bib,
                                    only_plot_results=only_plot_results)
        case 1:
            __dnn_train_autoencoder(config_dataset, config_train_ae, mode=1, noise_std=noise_std_ae,
                                    mode_cell_bib=mode_cell_bib,
                                    only_plot_results=only_plot_results)
        case 2:
            __dnn_train_autoencoder(config_dataset, config_train_ae, mode=2, noise_std=noise_std_ae,
                                    mode_cell_bib=mode_cell_bib,
                                    only_plot_results=only_plot_results)
        case 3:
            __dnn_train_combi(config_dataset, config_train_ae, config_train_class,
                              use_cell_bib=True, mode_cell_bib=mode_cell_bib, noise_std=noise_std_ae, do_plot=True)

        case 4:
            __dnn_train_classification(config_dataset, config_train_rgc,
                                       mode_cell_bib=mode_cell_bib,
                                       only_plot_results=only_plot_results)
        case 5:
            __dnn_train_spike_detection(config_dataset, config_train_class,
                                        only_plot_results=only_plot_results)
        case _:
            print("Wrong model!")
