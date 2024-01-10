import matplotlib.pyplot as plt
from package.plotting.plot_dnn import plot_statistic_data, results_training
from package.plotting.plot_metric import plot_loss, plot_confusion
from settings_ai import config_train_class, config_train_ae, config_data
from package.dnn.pytorch_classification import train_nn_classification
from package.dnn.pytorch_autoencoder import train_nn_autoencoder
from package.dnn.dataset.autoencoder_class import prepare_training as prepare_training_class
from package.dnn.dataset.autoencoder import prepare_training as prepare_training_ae


noise_std = 1
use_cell_bib = False
mode_cell_bib = 0
do_plot = False

# --- Main program
if __name__ == "__main__":
    plt.close("all")
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
    print("Training Autoencoder started")

    metric_snr_run = list()
    # ----------- Step #1: TRAINING AUTOENCODER
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = prepare_training_ae(data_settings=config_data,
                                  use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                  noise_std=noise_std)

    trainhandler = train_nn_autoencoder(config_train=config_train_ae, config_dataset=config_data)
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
        plt.show(block=True)

    del dataset, trainhandler
    print("Training Autoencoder ended")

    # ----------- Step #2: TRAINING CLASSIFIER
    # --- Processing: Loading dataset and Do Classification
    dataset = prepare_training_class(data_settings=config_data,
                                     path2model=path2model,
                                     use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                     noise_std=noise_std)
    trainhandler = train_nn_classification(config_train=config_train_class, config_dataset=config_data)
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
        plt.show(block=False)
    else:
        # --- Übergabe next run
        metric_snr_run.append((loss_ae, snr_ae, acc_class))

    del dataset, trainhandler
    print("ENDE")
