import numpy as np
import matplotlib.pyplot as plt
import src.plotting as pltSpAIke
import src_ai.processing_noise as aiprocess

from scipy.io import savemat

# from src_ai.nn_pytorch import NeuralNetwork
from src_ai.nn_tensorflow import NeuralNetwork

# TODO: Implement early break training modus
if __name__ == "__main__":
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # ----- Settings for AI -----
    NoFramesNoise = 1

    path2file = '0_checkMATLAB/denoising_dataset.npz'
    NoEpoch = 100
    SizeBatch = 32

    # ------ Loading Data
    print("... loading the datasets")

    npzfile = np.load(path2file)
    sorted(npzfile.files)

    frames_in = npzfile['arr_0']
    frames_mean = npzfile['arr_1']
    frames_cluster = npzfile['arr_2']

    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    # TODO: Adding noise to spike frames from datasets
    # --- Preparing data for training and validation
    frames_out = np.zeros(shape=frames_in.shape)

    for idx in range(0, frames_in.shape[0]):
        frames_out[idx] = frames_mean[[int(frames_cluster[idx, 0])]]

    # --- Generate fake frames with noise
    (noise_framesIn, noise_framesOut) = aiprocess.generate_frames(no_frames=NoFramesNoise, width_frames=frames_in.shape[1])

    TrainDataIn = np.concatenate((frames_in, noise_framesIn), axis=0)
    TrainDataOut = np.concatenate((frames_out, noise_framesOut), axis=0)
    print("... datasets for training are available")

    # --- Preparing PyTorch for Training
    nnTorch = NeuralNetwork(input_size=TrainDataIn.shape[1])
    nnTorch.initTrain(
        train_size=0.7,
        valid_size=0.2,
        shuffle=True,
        model_name="denoising_autoencoder_v0"
    )
    nnTorch.load_data(
        input=TrainDataIn, output=TrainDataOut,
        do_norm=True
    )

    # --- Training phase and Saving the model
    nnTorch.print_model()
    nnTorch.train_model(
        batch_size=SizeBatch,
        epochs=NoEpoch,
        learning_rate=1e-5
    )
    nnTorch.save_results(True)

    # --- Predicting results
    (Yin, Yout) = nnTorch.get_train_data()
    y_pred = nnTorch.predict_model(Yin)

    # --- Plotting
    pltSpAIke.plot_frames(TrainDataIn, TrainDataOut)
    pltSpAIke.plot_frames(Yin, y_pred)
    plt.show(block=True)

    # --- Saving data for MATLAB
    matdata = {"Train_Input": TrainDataIn, "Train_Output": TrainDataOut, "YPredIn": Yin, "YPredOut": y_pred}
    savemat("0_checkMATLAB/Data.mat", matdata)

    print()
    print("This is the End, ... my only friend, ... the end")
