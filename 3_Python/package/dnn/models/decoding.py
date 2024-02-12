from torch import nn, Tensor, unsqueeze, argmax
from package.dnn.pytorch_control import Config_PyTorch, Config_Dataset


# TODO: Modelle implementieren
class cnn_lstm_dec_v1(nn.Module):
    """Class of a convolutional Decoding for feature extraction"""

    def __init__(self):
        super().__init__()
        self.out_modelname = 'cnn_lstm_dec_v1'
        self.out_modeltyp = 'Decoder'
        self.model_embedded = False
        self.model_shape = (10, 10, 2)
        do_bias_train = True
        kernel_layer = [2, 10 , 32, 1]

        kernel_stride = [2, 2, 1]
        kernel_padding = [0, 0, 0]
        kernel_out = [0, 0, 0]

        self.cnn_1 = nn.Sequential(

            nn.Conv2d(kernel_layer[0], kernel_layer[1], kernel_size=(3,3) ,stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.ReLU(),

            nn.Conv2d(kernel_layer[1], kernel_layer[2],(3,3),stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.ReLU(),

        )

        self.flatten = nn.Flatten(start_dim=1)

        self.lstm_decoder = nn.Sequential(

        #nn.LSTMCell()
        )
    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        #x0 = unsqueeze(x, dim=1)
        cnnFeatuers = self.cnn_1(x)
        #encoded = self.encoder_linear(x0)
        #decoded0 = self.decoder_linear(encoded)
        #decoded0 = unsqueeze(decoded0, dim=1)
        #decoded = self.decoder(decoded0)

        return self.flatten(cnnFeatuers)


config_train_dec = Config_PyTorch(
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
Recommended_Config_PytorchSettings = Config_PyTorch(
    model=None,
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=40,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.25
)

Recommended_Config_DatasetSettings = Config_Dataset(
    # --- Settings of Datasets
    data_path='C:/Users/Muskelgurke/Downloads/',
    data_file_name='2024-02-05_Dataset-KlaesNeuralDecoding.npy',
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=0,
    # --- Dataset Preparation
    data_exclude_cluster=[],
    data_sel_pos=[]
)
