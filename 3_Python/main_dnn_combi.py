import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy.io import loadmat
from package.plotting.plot_dnn import plot_statistic_data
from package.plotting.plot_metric import plot_loss, prep_confusion
from settings_ai import config_train_class_pytorch, config_train_ae_pytorch, config_train_class_dataset, config_train_ae_dataset
from package.dnn.pytorch_classification import train_nn_classification
from package.dnn.pytorch_autoencoder import train_nn_autoencoder
from package.dnn.dataset.autoencoder_class import prepare_training as prepare_training_class
from package.dnn.dataset.autoencoder import prepare_training as prepare_training_ae


noise_std = 1
use_cell_bib = False
mode_cell_bib = 0

# --- Main program
if __name__ == "__main__":
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
    print("Training Autoencoder started")
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = prepare_training_ae(path2data=config_train_ae_dataset.get_path2data(), data_settings=config_train_ae_dataset,
                                  use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                  noise_std=noise_std)
    data_mean = dataset.frames_me
    trainhandler = train_nn_autoencoder(config_train=config_train_ae_pytorch, config_dataset=config_train_ae_dataset)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    snr = trainhandler.do_training()
    data_result_ae = trainhandler.do_validation_after_training()
    path2model = trainhandler.get_saving_path()
    print("Training Autoencoder ended")
    del dataset, data_mean, trainhandler

    # --- Processing: Loading dataset and Do Classification
    dataset = prepare_training_class(path2data=config_train_class_dataset.get_path2data(), data_settings=config_train_class_dataset,
                                     path2model=path2model,
                                     use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                     noise_std=noise_std)
    data_mean = dataset.frames_me
    trainhandler = train_nn_classification(config_train=config_train_class_pytorch, config_dataset=config_train_class_dataset)
    trainhandler.load_model()
    trainhandler.load_data(dataset)

    epoch_acc = trainhandler.do_training()[0]
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()

    del dataset

    plt.close('all')
    # plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir, epoch_zoom=[80, ])
    prep_confusion(data_result['valid_clus'], data_result['yclus'], "training", "both", False,
                   cl_dict=data_result['cl_dict'], path2save=logsdir)
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])

    plt.show(block=False)