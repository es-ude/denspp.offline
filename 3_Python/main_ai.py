import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from settings import Settings
from src.call_data import call_data
from src.afe import AFE
import src.plotting as pltSpAIke
import src_ai.processing as aiprocess

#from src_ai.nn_pytorch import NeuralNetwork
from src_ai.nn_tensorflow import NeuralNetwork

# TODO: Implement early break training modus

if __name__ == "__main__":
    print("Train modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    settings = Settings()
    afe = AFE(settings)
    # ----- Settings for AI -----
    # TakeDatasets = np.array([1, 2, 3])
    # NoDataPoints = np.array([5, 10, 23])
    TakeDatasets = np.array([3])
    NoDataPoints = np.array([1,2,3])

    NoFramesNoise = 1
    NoEpoch = 50
    SizeBatch = 16

    # ------ Loading Data
    frames_in = []
    frames_mean = []
    frames_cluster = []
    runs = 0
    print("... loading the datasets")
    for runSet in TakeDatasets:
        runPoint = 1
        while(runPoint <= NoDataPoints[runSet-1]):
            (neuron, labeling) = call_data(
                path2data=settings.path2data,
                data_type=runSet,
                data_set=runPoint,
                desired_fs=settings.desired_fs,
                t_range=settings.t_range,
                ch_sel=settings.ch_sel,
                plot=False
            )

            # Adapting the input for our application
            u_ana, _ = afe.pre_amp(neuron.data)
            x_adc = afe.adc_nyquist(u_ana, 1)

            x_pos = np.floor(labeling.spike_xpos * settings.sample_rate/neuron.fs).astype("int")

            (frame_raw, frame_mean, frame_cluster) = aiprocess.getting_frames(
                data=x_adc, xpos=x_pos,
                dxneg=-10, dxpos=10+afe.frame_length,
                cluster=labeling.cluster_id,
                cluster_no=frames_cluster
            )
            if runs == 0:
                frames_in = frame_raw
                frames_mean = frame_mean
                frames_cluster = frame_cluster
            else:
                frames_in = np.concatenate((frames_in, frame_raw), axis=0)
                frames_mean = np.concatenate((frames_mean, frame_mean), axis=0)
                frames_cluster = np.concatenate((frames_cluster, frame_cluster), axis=0)

            runs += 1
            runPoint += 1

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
    nnTorch = NeuralNetwork(
        train_size=0.7, valid_size=0.2,
        shuffle=True,
        input_size=TrainDataIn.shape[1],
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
    savemat("Data.mat", matdata)

    print()
    print("This is the End, ... my only friend, ... the end")
