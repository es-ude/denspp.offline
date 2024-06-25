from torch import nn
import matplotlib.pyplot as plt
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import package.dnn.models.decoding_utah as models_dec

config_data = Config_Dataset(
    # --- Settings of Datasets
    # data_path='/home/muskel/Documents/cpsDEC/data', # Ubuntu
    data_path='C:\spaikeDenSppDataset', # Ubuntu
    data_file_name='2024-02-05_Dataset-KlaesNeuralDecoding.npy',

    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=1000,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='CPU',
    data_normalization_method='minmax',
    data_normalization_setting='bipolar',
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_dec.cnn_lstm_dec_v3(1, 12, 3),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def do_train_decoder_utah(dnn_handler: dnn_handler, length_window_ms=500) -> None:
    """Training routine for Neural Decoding of recordings from Utah array (KlaesLab)
    Args:
        dnn_handler: Handler for configurating the routine selection for train deep neural networks
        length_window_ms: Size of the time window for segmenting the tick interval into firing events
    """
    from package.dnn.dataset.decoding_utah import prepare_training
    from package.dnn.pytorch.rnn import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)")

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(config_data, length_window_ms)
    data_dict = dataset.frame_dict
    num_output = len(data_dict)
    trainhandler = train_nn(config_train, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training(num_output)
    logsdir = trainhandler.get_saving_path()
    del trainhandler

    # --- Plotting and Ending
    if dnn_handler.do_plot:
        plt.close("all")
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=data_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir, cl_dict=data_dict)

        plt.show(block=dnn_handler.do_block)

    print("\nThe End")
