import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from torch import nn, from_numpy, load
from scipy.io import savemat

from package.plotting.plot_metric import plot_confusion, plot_loss
from package.plotting.plot_dnn import plot_statistic_data
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.rgc_onoff_class import prepare_plotting, prepare_training
import package.dnn.models.rgc_onoff_class as ai_module


config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_rgc_v1(output_size=2),
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=1000,
    batch_size=128,
    # --- Settings of Datasets
    data_path='data',
    data_file_name='2023-11-16_rgc_onoff_fzj.mat',
    # data_file_name='2023-11-17_Dataset-07_RGC_TDB_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=0,
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
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train,
                               reduce_fzj_data=True, reduce_rgc_data=False)
    dataset_dict = dataset.frame_dict if dataset.cluster_name_available else []
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_metric = trainhandler.do_training()[1]

    # --- Post-Processing: Getting data from validation set for inference
    xdata, xclus = prepare_plotting(trainhandler.train_loader)
    ydata, yclus = prepare_plotting(trainhandler.valid_loader)
    xdata0 = np.append(xdata, ydata, axis=0)
    xclus0 = np.append(xclus, yclus, axis=0)
    del xdata, ydata

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
             "config": config_train},
            do_compression=True,
            long_field_names=True)

    # --- Plotting
    plot_loss(epoch_metric, 'Acc.', path2save=logsdir)
    plot_confusion(xclus0, ypred, path2save=logsdir, cl_dict=dataset_dict)
    plot_statistic_data(xclus, yclus, path2save=logsdir, cl_dict=dataset_dict)

    plt.show(block=False)
    plt.close("all")
