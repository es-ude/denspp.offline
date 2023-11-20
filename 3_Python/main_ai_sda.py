import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from torch import nn, from_numpy, load
from scipy.io import savemat

from package.plotting.plot_metric import plot_confusion, plot_loss
from package.plotting.plot_dnn import plot_statistic_data
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.spike_detection import prepare_plotting, prepare_training
import package.dnn.models.spike_detection as ai_module


config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_sda_v1(),
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    # --- Settings of Datasets
    data_path='data',
    data_file_name='SDA_Dataset.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Settings for Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=2000,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=True,
    data_num_samples_per_cluster=20000,
    # --- Dataset Preparation
    data_exclude_cluster=[],
    data_sel_pos=[]
)

# --- Hauptprogramm
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train, threshold=2)
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    loss, epoch_metric = trainhandler.do_training()

    # --- Post-Processing: Getting data from validation set for inference
    xdata, xsda, xclus = prepare_plotting(trainhandler.train_loader)
    ydata, ysda, yclus = prepare_plotting(trainhandler.valid_loader)
    xdata0 = np.append(xdata, ydata, axis=0)
    xsda0 = np.append(xsda, ysda, axis=0)
    xclus0 = np.append(xclus, yclus, axis=0)
    del xdata, ydata, xsda, ysda

    # --- Post-Processing: Do the Inference with Best Model
    print(f"\nDoing the inference with validation data on best model")
    model_inference = load(trainhandler.get_best_model()[0])
    ypred = model_inference(from_numpy(xdata0))[1]
    ypred = ypred.detach().numpy()

    # --- Saving results
    logsdir = trainhandler.get_saving_path()
    savemat(join(logsdir, 'results.mat'),
            {"frames_in": xdata0,
             "cluster_orig": xclus0,
             "cluster_pred": ypred,
             "config": config_train
             },
            do_compression=True,
            long_field_names=True)

    # --- Plotting
    plot_loss(epoch_metric, 'Acc.', path2save=logsdir)
    plot_confusion(xclus0, ypred, path2save=logsdir)
    plot_statistic_data(xclus, yclus, path2save=logsdir)

    plt.show(block=False)
    plt.close("all")

    print("\nLook data on TensorBoard -> open Terminal")
    print("Type in: tensorboard serve --logdir ./runs")
