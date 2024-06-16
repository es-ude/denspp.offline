from torch import nn, Tensor, argmax
import torch
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset

class cnn_lstm_dec_v2(nn.Module):
    """Class of a convolutional Decoding for feature extraction"""
    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__()
        self.out_modelname = 'cnn_lstm_dec_v1'
        self.out_modeltyp = 'Decoder'
        self.model_embedded = False
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        do_bias_train = True
        # --- Settings for CNN
        # --- Settings for CNN
        kernel_layer = [num_clusters, 10, 20]
        kernel_stride = [3, 3, 2]  # how much we move
        kernel_padding = [0, 0, 0]
        # --- Settings for DNN/LSTM
        dense_layer_size = [40, 32, output_samples]

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(in_channels = num_clusters,
                      out_channels= kernel_layer[1],
                      kernel_size=3,
                      stride= 1,
                      padding= 0),
            nn.BatchNorm2d(num_features= kernel_layer[1]),
            nn.ReLU(),
        )
        self.dnn_1 = nn.Sequential(
            nn.Linear(dense_layer_size[0], dense_layer_size[1], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(dense_layer_size[1], dense_layer_size[2], bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[2]),
            nn.Softmax(dim=1)
        )

        self.lstm = nn.LSTM(input_size=dense_layer_size[2], hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, output_samples)

    def forward(self, x):
        batch_size, num_clusters, height, width, num_time_windows = x.shape
        cnn_outs = []

        for i in range(num_time_windows):
            img = x[:, :, :, :, i]
            cnn_out = self.cnn_1(img)
            cnn_out = cnn_out.view(batch_size, -1)  # Flatten
            cnn_out = self.dnn_1(cnn_out)
            cnn_outs.append(cnn_out.unsqueeze(1))  # Add time dimension
        print(cnn_out)
        lstm_input = torch.cat(cnn_outs, dim=1)  # Concatenate along the time dimension
        print(lstm_input)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.fc(lstm_out[:, -1, :])  # Take the output of the last LSTM cell

        return lstm_out

Recommended_Config_PytorchSettings = Config_PyTorch(
    # --- Settings of Models/Training
    model=cnn_lstm_dec_v2(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

Recommended_Config_DatasetSettings = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2024-02-05_Dataset-KlaesNeuralDecoding.npy',
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

    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__()
        self.out_modelname = 'cnn_lstm_dec_v1'
        self.out_modeltyp = 'Decoder'
        self.model_embedded = False
        self.model_shape = (1, num_clusters, 10, 10, input_samples)
        do_bias_train = True
        # --- Settings for CNN
        kernel_layer = [num_clusters, 10, 20]
        kernel_stride = [3, 3, 2] # how much we move
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
        print(cnn_feat)
        pred_con = self.dnn_1(cnn_feat)
        print(pred_con)
        return pred_con, argmax(pred_con, 1)


Recommended_Config_PytorchSettings = Config_PyTorch(
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

Recommended_Config_DatasetSettings = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2024-02-05_Dataset-KlaesNeuralDecoding.npy',
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
class cnn2D_v2(nn.Module):
    """Class of a 2D convolutional Decoding for feature extraction 06/2024"""
    def __init__(self, num_clusters=1, input_samples=12, output_samples=3):
        super().__init__()
        self.out_modelname = 'cnn2D_v2'
        self.out_modeltyp = 'Decoder'
        self.model_embedded = False # ist es auf einer Embedded Hardware?
        self.model_shape = (1, num_clusters, 10, 10, input_samples) # prepareTraining return Dataloader(...) Dimensionen
        self.input_samples = input_samples
        self.num_clusters = num_clusters
        self.output_samples = output_samples


        # --- Settings for CNN
        kernel_layer = [10, 20]
        kernel_stride = [3, 3, 2]
        kernel_padding = [0, 0, 0]
        do_bias_train = True
        # --- Settings for DNN/LSTM
        dense_layer_size = [40, 32, output_samples]

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_clusters,
                out_channels=10,
                kernel_size=(3,3),
                stride= 1,
                padding=0
            ),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3,3),
                stride= 1,
                padding=0
            ),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )
        self.dnn_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                dense_layer_size[0],
                dense_layer_size[1],
                bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[1]),
            nn.ReLU(),
            nn.Linear(
                dense_layer_size[1],
                dense_layer_size[2],
                bias=do_bias_train),
            nn.BatchNorm1d(dense_layer_size[2]),
            nn.Softmax()
        )

        self.flatten = nn.Flatten()


        self.lstm_decoder = nn.Sequential(
            #nn.LSTMCell()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        batch_size = x.size(0) # batchSize ist immer am Anfang zu finden
        x = x.view(batch_size*self.input_samples, self.num_clusters, 10, 10)

        print("debugging <3")


        return pred_con, argmax(pred_con, 1)

class cnn2D_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_modelname = 'cnn_ae_v1_2d'
        self.out_modeltyp = 'Autoencoder'
        self.model_embedded = False
        self.model_shape = (1, 32)
        do_bias_train = True
        kernel_layer = [1, 22, 8, 3]
        kernel_size = [(4, 4), (3, 3), (3, 3)]  # 2D kernel size
        kernel_stride = [(2, 2), (2, 2), (2, 2)]  # 2D kernel stride
        kernel_padding = [(0, 0), (0, 0), (0, 0)]  # 2D kernel padding
        kernel_out = [(0, 0), (0, 0), (0, 0)]  # 2D kernel output padding

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm2d(kernel_layer[1], affine=do_bias_train),
            nn.ReLU(),
            nn.Conv2d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm2d(kernel_layer[2], affine=do_bias_train),
            nn.ReLU(),
            nn.Conv2d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm2d(kernel_layer[3], affine=do_bias_train),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kernel_layer[3], kernel_layer[2], kernel_size[2], stride=kernel_stride[2],
                               padding=kernel_padding[2], output_padding=kernel_out[2]),
            nn.BatchNorm2d(kernel_layer[2], affine=do_bias_train),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_layer[2], kernel_layer[1], kernel_size[1], stride=kernel_stride[1],
                               padding=kernel_padding[1], output_padding=kernel_out[1]),
            nn.BatchNorm2d(kernel_layer[1], affine=do_bias_train),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_layer[1], kernel_layer[0], kernel_size[0], stride=kernel_stride[0],
                               padding=kernel_padding[0], output_padding=kernel_out[0]),
            nn.BatchNorm2d(kernel_layer[0], affine=do_bias_train),
            nn.ReLU(),
            nn.Linear(24, self.model_shape[1], bias=True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.flatten(encoded), self.flatten(decoded)


