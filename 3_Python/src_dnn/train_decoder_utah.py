from torch import nn
import matplotlib.pyplot as plt
import os
from package.dnn.dnn_handler import dnn_handler  #  hier hab ich nie was geändert, kann aus PACKAGE importiert werden!!
from src_dnn.src_pytorch_handler import ConfigPyTorch, ConfigDataset
import src_dnn.models.src_models_decoding_utah as models_decoding
from pathlib import Path

config_data = ConfigDataset(
    # --- Settings of Datasets
    data_path='/home/muskel/Documents/cpsDEC/data',  # Ubuntu
    #data_path='C:\\spaikeDenSppDataset',
    data_file_name='2024-02-05_Dataset-KlaesNeuralDecoding.npy',

    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='CPU',
    data_normalization_method='minmax',
    data_normalization_setting='bipolar',
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster= 0,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

ConfigTrain = ConfigPyTorch(
    # --- Settings of Models/Training
    model=models_decoding.test_model_if_pipeline_running(1, 12, 3),
    #model=models_decoding.cnn_lstm_dec_v3(1, 12, 3),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    data_split_ratio=0.25,
    data_do_shuffle=False   ,
    deterministic_training=False,
    seed = 133
)


def do_train_decoder_utah(dnn_handler: dnn_handler, length_window_ms=500) -> None:
    # ToDo -> dnn_handler übergibt nur Einstellungen --> in Settings.yaml übertragen
    """Training routine for Neural Decoding of recordings from Utah array (KlaesLab)
    Args:
        dnn_handler: Handler for configurating the routine selection for train deep neural networks
        length_window_ms: Size of the time window for segmenting the tick interval into firing events
    """
    from src_dnn.dataset.src_dataset_decoding_utah import preprocess_dataset
    from src_dnn.pytorch.src_lstm_Decoding import TrainHandlerLstm  # Import der pytorch file
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    base_path = Path(__file__).parents[2]
    func_name = do_train_decoder_utah.__name__
    # Pfad ab dem Ordner "3_Python" extrahieren
    shortened_path = Path(__file__).relative_to(base_path)
    print(
        f"\n\n=== Executing function --> {func_name} in file --> {shortened_path} ===")
    print("\n\t Train modules of end-to-end neural signal pre-processing frame-work (DeNSPP)")

    # --- Processing: Loading Data
    dataset = preprocess_dataset(config_data, length_window_ms)
    data_deci_label = dataset.lable_dict
    num_output = len(data_deci_label)

    trainhandler = TrainHandlerLstm(ConfigTrain, config_data)
    trainhandler.preperation_for_training()
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
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=data_deci_label)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir,
                            cl_dict=data_deci_label)

        #plt.show(block=dnn_handler.do_block)

    print("\nThe End")
