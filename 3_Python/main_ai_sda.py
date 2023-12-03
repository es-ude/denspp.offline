import matplotlib.pyplot as plt
from torch import nn
from package.plotting.plot_dnn import plot_statistic_data
from package.plotting.plot_metric import plot_confusion, plot_loss
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.spike_detection import prepare_training
import package.dnn.models.spike_detection as ai_module


num_output = 2

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_sda_v1(input_size=12, output_size=num_output),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=5,
    batch_size=64,
    # --- Settings of Datasets
    data_path='data',
    data_file_name='2023-11-21_SDA_Dataset.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Settings for Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=2000,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=True,
    data_num_samples_per_cluster=5000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

# --- Main Program
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train,
                               threshold=4)
    dataset_dict = dataset.sda_dict
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[0]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training(3)

    logsdir = trainhandler.get_saving_path()
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=dataset_dict)
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir, cl_dict=dataset_dict)

    plt.show(block=False)
    plt.close("all")
