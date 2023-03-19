import os.path

import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import src.plotting as pltSpAIke
from src_ai.processing_noise import generate_noiseframe
from src_ai.processing_data import prepare_data_ae_training

from scipy.io import savemat, loadmat
# from src_ai.nn_pytorch import NeuralNetwork
# from src_ai.nnpy_architecture import dnn_autoencoder as nn_network
from src_ai.nn_tensorflow import NeuralNetwork
from src_ai.nntf_architecture import cnn_autoencoder as nn_network

# TODO: Implement early break training modus
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # ----- Settings for AI -----
    path2data = 'src_ai/data'
    file_name = '2023-03-16_Martinez2009_File1_Sorted.mat'
    do_addnoise = False
    NoFramesNoise = 1
    # Setzen des Ignorier-Clusters in Line 89

    # ------ Loading Data
    datumHeute = date.today()
    str_datum = datumHeute.strftime("%Y%m%d-%H%M%S")
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    data_type = False
    path2file = os.path.join(path2data, file_name)
    if file_name[-3:] == "npz":
        # --- NPZ reading file
        npzfile = np.load(path2file)
        frames_in = npzfile['arr_0']
        frames_cluster = npzfile['arr_2']
    else:
        # --- MATLAB reading file
        npzfile = loadmat(path2file)
        frames_in = npzfile["frames_in"]
        frames_cluster = npzfile["frames_cluster"]

    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    # --- Calculation of the mean waveform
    NoCluster = np.unique(frames_cluster)
    SizeCluster = np.size(NoCluster)
    SizeFrame = frames_in.shape
    frames_mean = np.zeros(shape=(SizeCluster, SizeFrame[1]), dtype=float)

    for idx in NoCluster:
        selX = np.where(frames_cluster == idx)
        frames_sel = frames_in[selX[0], :]
        frames_mean[idx-1, :] = np.mean(frames_sel, axis=0, dtype=int)

    # --- Preparing data for training and validation
    # Step 1: Building the mean waveforms
    corValue = np.argmin(frames_cluster)
    frames_out = np.zeros(shape=frames_in.shape)
    for idx in range(0, frames_in.shape[0]-corValue):
        frames_out[idx] = frames_mean[[int(frames_cluster[idx]-1)]]

    # TODO: Adding noise to spike frames from datasets (adding fake frames?)
    # Step 2: Adding generated noise to the input
    if do_addnoise:
        (noise_framesIn, noise_framesOut) = generate_noiseframe(
            no_frames=NoFramesNoise,
            width_frames=frames_in.shape[1]
        )
        TrainDataIn = np.concatenate((frames_in, noise_framesIn), axis=0)
        TrainDataOut = np.concatenate((frames_out, noise_framesOut), axis=0)
    else:
        TrainDataIn = frames_in
        TrainDataOut = frames_out

    # TODO: MinMaxScaler kann positive sein --> Testen
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = np.vstack((TrainDataIn, TrainDataOut))
    # data = scaler.fit_transform(data)
    # TrainDataIn = data[0:len(TrainDataIn)][:]
    # TrainDataOut = data[len(TrainDataIn):][:]

    # Step 3: Splitting data for training of the denoising autoencoder
    (Xin, Yin, cluster_in, Xout, Yout, cluster_out) = prepare_data_ae_training(
        TrainDataIn, TrainDataOut,
        cluster=frames_cluster, do_cluster=NoCluster[1:],
        train_size=0.7, valid_size=0.2
    )

    print("... datasets for training are available")

    # --- Preparing PyTorch network
    nnTorch = NeuralNetwork()
    nnTorch.defineModel(
        model=nn_network(),
        input_size=TrainDataIn.shape[1]
    )
    nnTorch.initTrain(
        train_size=0.7, valid_size=0.2,
        shuffle=False,
        name_addon=""
    )
    # --- Loading data
    nnTorch.load_data_direct(
        train_in=Xin, train_out=Xout,
        valid_in=Yin, valid_out=Yout,
        do_norm=True
    )
    # --- Training phase and Saving the model
    nnTorch.print_model()
    nnTorch.do_training()
    nnTorch.save_results()

    # --- Predicting results
    (TrainIn, TrainOut, ValidIn, ValidOut) = nnTorch.get_train_data()
    print("... do validation of network")
    feat, y_pred = nnTorch.do_prediction(ValidIn)

    # --- Saving data for MATLAB (Wichtig: NumPy Arrays zum Übertragen)
    print("")
    matdata = {"Train_Input": TrainIn.numpy(), "Train_Output": TrainOut.numpy(), "PredIn": ValidIn.numpy(), "PredOut": ValidOut.numpy(), "YPred": y_pred.numpy(), "Feat": feat.numpy(), "Cluster": cluster_out}
    savemat("logs/" + str_datum + "_" + nnTorch.set_name + ".mat", matdata)

    # --- Plotting
    pltSpAIke.plot_frames(TrainDataIn, TrainDataOut)
    pltSpAIke.plot_frames(Yin, y_pred)
    plt.show(block=False)

    print("\nThis is the End, ... my only friend, ... the end")
