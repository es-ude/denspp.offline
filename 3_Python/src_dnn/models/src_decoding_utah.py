from torch import nn, Tensor, argmax
from package.dnn.pytorch_handler import ConfigPyTorch, ConfigDataset


class cnn_rnn_v2(nn.Module):
    """Class of a convolutional Decoding for feature extraction"""
    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__()
        self.out_modelname = 'cnn_rnn_v2'
        self.out_modeltyp = 'Decoder'
        self.model_embedded = False
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        do_bias_train = True
        # --- Settings for CNN
        kernel_layer = [num_clusters, 10, 20]
        kernel_stride = [3, 3, 2]
        kernel_padding = [0, 0, 0]
        # --- Settings for DNN/LSTM
        dense_layer_size = [40, 32, output_samples]

        self.cnn_1 = nn.Sequential(
            nn.Conv3d(kernel_layer[0], kernel_layer[1], kernel_size=(3, 3, 1),
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm3d(kernel_layer[1]),
            nn.ReLU(),
            nn.Conv3d(kernel_layer[1], kernel_layer[2], kernel_size=(3, 3, 1),
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm3d(kernel_layer[2]),
            nn.ReLU()
        )
        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_layer_size[0], dense_layer_size[1], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(dense_layer_size[1], dense_layer_size[2], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[2]),
            nn.Softmax()
        )

        self.flatten = nn.Flatten(start_dim=0)
        self.lstm_decoder = nn.Sequential(
            #nn.LSTMCell()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        cnn_feat = self.cnn_1(x)
        pred_con = self.dnn_1(cnn_feat)
        return pred_con, argmax(pred_con, 1)


Recommended_Config_PytorchSettings = ConfigPyTorch(
    # --- Settings of Models/Training
    model=cnn_rnn_v2(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

Recommended_Config_DatasetSettings = ConfigDataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='',
    data_normalization_method='',
    data_normalization_setting='',
    # --- Dataset Preparation
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=0,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

class cnn_lstm_dec_v1(nn.Module):
    """Class of a convolutional Decoding for feature extraction"""

    def __init__(self, num_clusters=1, input_samples=12, output_sampes=3):
        super().__init__()
        self.out_modelname = 'cnn_lstm_dec_v1'
        self.out_modeltyp = 'Decoder'
        self.model_embedded = False
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        do_bias_train = True
        # --- Settings for CNN
        kernel_layer = [num_clusters, 10, 20]
        kernel_stride = [3, 3, 2]
        kernel_padding = [0, 0, 0]
        # --- Settings for DNN/LSTM
        dense_layer_size = [40, 32, output_sampes]

        self.cnn_1 = nn.Sequential(
            nn.Conv3d(kernel_layer[0], kernel_layer[1], kernel_size=(3, 3, 1),
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm3d(kernel_layer[1]),
            nn.ReLU(),
            nn.Conv3d(kernel_layer[1], kernel_layer[2], kernel_size=(3, 3, 1),
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm3d(kernel_layer[2]),
            nn.ReLU()
        )
        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_layer_size[0], dense_layer_size[1], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(dense_layer_size[1], dense_layer_size[2], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[2]),
            nn.Softmax()
        )

        self.flatten = nn.Flatten(start_dim=0)
        self.lstm_decoder = nn.Sequential(
            #nn.LSTMCell()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        cnn_feat = self.cnn_1(x)
        pred_con = self.dnn_1(cnn_feat)
        return pred_con, argmax(pred_con, 1)


Recommended_Config_PytorchSettings = ConfigPyTorch(
    # --- Settings of Models/Training
    model=cnn_lstm_dec_v1(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

Recommended_Config_DatasetSettings = ConfigDataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='',
    data_normalization_method='',
    data_normalization_setting='',
    # --- Dataset Preparation
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=0,
    data_exclude_cluster=[],
    data_sel_pos=[]
)
