import matplotlib.pyplot as plt
import numpy as np

from settings_ai import config_train_ae, config_dataset
from package.dnn.pytorch.autoencoder import train_nn
from package.dnn.dataset.autoencoder import prepare_training as prepare_training_ae, DatasetAE

from package.metric import calculate_snr


noise_std = 0
use_cell_bib = False
mode_cell_bib = 0
do_plot = False

snr_soll = np.linspace(-10, 10, 21, endpoint=True)
noise_std_increasing_step = 1.0
STD_SNR_ratio_plotting_data = []


def calculate_dataset_snr(dataset: DatasetAE) -> [list, list]:
    snr0 = [[] for _ in np.unique(dataset.cluster_id)]
    for data in dataset:
        snr0[data["cluster"]].append(calculate_snr(data["in"], data["mean"]))

    snr1 = list()
    for snr_val in snr0:
        snr_id = np.array(snr_val)
        snr1.append((snr_id.min(), np.median(snr_id), snr_id.max()))

    return snr0, snr1


def prepare_dataset_snr(dataset: DatasetAE, snr_in: float) -> DatasetAE:
    """Generating a new dataset with mean waveform"""
    frames_mean = dataset.frames_me
    cluster_sel = dataset.cluster_id

    frames_new = np.zeros((cluster_sel.size, frames_mean.shape[1]))
    for idx, id in enumerate(cluster_sel):
        frames_new[idx, :] = frames_mean[id, :]

    return DatasetAE(frames_new, cluster_sel, frames_mean,
                     cluster_dict=dataset.frame_dict, mode_train=dataset.mode_train)


if __name__ == "__main__":
    plt.close("all")
    # Getting the data (once)
    dataset = prepare_training_ae(path2data=config_dataset.get_path2data(), data_settings=config_dataset,
                                  use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                  noise_std=noise_std)
    snr_in_raw, snr_in_val = calculate_dataset_snr(dataset)

    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
    metric_snr_run = list()
    # --- Run with increasing SNR
    for run_id, snr in enumerate(snr_soll):
        print(f"... training Autoencoder: Run {run_id} started")
        # ----------- Step #1: TRAINING AUTOENCODER
        # --- Processing: Loading dataset
        dataset_used = prepare_dataset_snr(dataset)

        # --- Do Autoencoder Training
        trainhandler = train_nn(config_train=config_train_ae, config_dataset=config_dataset)
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
