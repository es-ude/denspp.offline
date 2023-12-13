import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy.io import loadmat
from package.plotting.plot_dnn import plot_statistic_data
from package.plotting.plot_metric import plot_loss, plot_confusion
from settings_ai import config_train_class as config_train
from package.dnn.pytorch_classification import train_nn_classification
from package.dnn.dataset.autoencoder_class import prepare_training


noise_std = 1
use_cell_bib = False
mode_cell_bib = 0
only_plot = False

# --- Main program
if __name__ == "__main__":
    if not only_plot:
        print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
        # --- Processing: Loading dataset and Do Training
        dataset = prepare_training(path2data=config_train.get_path2data(), settings=config_train,
                                   path2model='runs/20231213_171117_train_cnn_ae_v4',
                                   use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                   noise_std=noise_std)
        data_mean = dataset.frames_me
        trainhandler = train_nn_classification(config_train)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset

        epoch_acc = trainhandler.do_training()[0]
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()
    else:
        epoch_acc = list()
        logsdir = 'runs/20231204_152721_train_cnn_ae_v3/'
        data_result = loadmat(join(logsdir, 'results.mat'))
        data_mean = np.zeros(shape=(52, 32))

    plt.close('all')
    # plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir, epoch_zoom=[80, ])
    plot_confusion(data_result['valid_clus'], data_result['yclus'],
                   path2save=logsdir, cl_dict=data_result['cl_dict'])
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])

    plt.show(block=False)