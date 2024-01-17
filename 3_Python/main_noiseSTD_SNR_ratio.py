import matplotlib.pyplot as plt
import csv, os
from package.plotting.plot_dnn import plot_statistic_data, results_training
from settings_ai import config_train_class, config_train_ae, config_dataset
from package.dnn.pytorch_autoencoder import train_nn_autoencoder
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.dataset.autoencoder_class import prepare_training as prepare_training_class
from package.dnn.dataset.autoencoder import prepare_training as prepare_training_ae


noise_std = 0
use_cell_bib = False
mode_cell_bib = 0
do_plot = False
noise_std_increasing_step = 1.0
STD_SNR_ratio_plotting_data = []


def generate_dataset_snr(dataset):
    print("Test")
    frames_mean = dataset.frames_me

    for val in dataset:
        print(val)

    return dataset


def prepare_dataset_snr(dataset):
    return dataset


if __name__ == "__main__":
    plt.close("all")
    # Getting the data (once)
    dataset = prepare_training_ae(path2data=config_dataset.get_path2data(), data_settings=config_dataset,
                                  use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                  noise_std=noise_std)
    dataset_zero = generate_dataset_snr(dataset)

    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
    metric_snr_run = list()
    # --- Run with increasing SNR
    for i in range(10):
        print(f"... training Autoencoder: Run {i} started")
        # ----------- Step #1: TRAINING AUTOENCODER
        # --- Processing: Loading dataset
        dataset_used = prepare_dataset_snr(dataset)

        # --- Do Autoencoder Training
        trainhandler = train_nn_autoencoder(config_train=config_train_ae, config_dataset=config_dataset)
        logsdir = trainhandler.get_saving_path()
        trainhandler.load_model()
        trainhandler.load_data(dataset_used)
        loss_ae, snr_ae = trainhandler.do_training()[-1]
        #path2model = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()

        # --- Übergabe next run
        metric_snr_run.append((loss_ae, snr_ae))

        STD_SNR_ratio_plotting_data.append(
            (data_result.get('median_SNR_in', None), data_result.get('median_SNR_inc', None))
        )

        # Loop control
        noise_std = noise_std + noise_std_increasing_step

    # --- Plotting the results
    plt.plot(*zip(*STD_SNR_ratio_plotting_data), marker='o')
    plt.xlabel('median SNR in')
    y_label = "Median SNR_Increase"
    plt.ylabel(y_label)
    plt.title('Noise standard Deviation Impact on Median Signal to Noise Ratio')
    plt.savefig(logsdir)
    plt.show()

del dataset, trainhandler
print("Ended")