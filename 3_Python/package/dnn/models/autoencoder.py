from torch import nn, Tensor, unsqueeze
from package.dnn.pytorch_control import Config_PyTorch


class dnn_ae_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.out_modelname = 'dnn_ae_v1'
        self.out_modeltyp = 'Autoencoder'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, 14, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[3], affine=do_train_batch),
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class dnn_ae_v2(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.out_modelname = 'dnn_ae_v2'
        self.out_modeltyp = 'Autoencoder'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class cnn_ae_v1(nn.Module):
    """Class of a convolutional autoencoder for feature extraction"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'cnn_ae_v1'
        self.out_modeltyp = 'Autoencoder'
        self.model_shape = (1, 32)
        self.model_embedded = False
        do_bias_train = True
        kernel_layer = [1, 16, 6, 1]
        kernel_size = [4, 3, 2]
        kernel_stride = [2, 2, 1]
        kernel_padding = [0, 0, 0]
        kernel_out = [0, 0, 0]

        self.encoder = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm1d(kernel_layer[3], affine=do_bias_train)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[3], kernel_layer[2], kernel_size[2], stride=kernel_stride[2],
                               padding=kernel_padding[2], output_padding=kernel_out[2]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[2], kernel_layer[1], kernel_size[1], stride=kernel_stride[1],
                               padding=kernel_padding[1], output_padding=kernel_out[1]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[1], kernel_layer[0], kernel_size[0], stride=kernel_stride[0],
                               padding=kernel_padding[0], output_padding=kernel_out[0])
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x0 = unsqueeze(x, dim=1)
        encoded = self.encoder(x0)
        decoded = self.decoder(encoded)
        return self.flatten(encoded), self.flatten(decoded)


class cnn_ae_v2(nn.Module):
    """Class of a convolutional autoencoder for feature extraction"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'cnn_ae_v2'
        self.out_modeltyp = 'Autoencoder'
        self.model_embedded = False
        self.model_shape = (1, 32)
        do_bias_train = True
        kernel_layer = [1, 22, 8, 3]
        kernel_size = [4, 3, 3]
        kernel_stride = [2, 2, 2]
        kernel_padding = [0, 0, 0]
        kernel_out = [0, 0, 0]
        pool_size = [2, 2]
        pool_stride = [2, 2]

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm1d(kernel_layer[3], affine=do_bias_train)
        )
        self.pool = nn.MaxPool1d(pool_size[0], stride=pool_stride[0], return_indices=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.un_pool = nn.MaxUnpool1d(pool_size[1], stride=pool_stride[1])
        # Decoder setup
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[3], kernel_layer[2], kernel_size[2], stride=kernel_stride[2],
                               padding=kernel_padding[2], output_padding=kernel_out[2]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[2], kernel_layer[1], kernel_size[1], stride=kernel_stride[1],
                               padding=kernel_padding[1], output_padding=kernel_out[1]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[1], kernel_layer[0], kernel_size[0], stride=kernel_stride[0],
                               padding=kernel_padding[0], output_padding=kernel_out[0]),
            nn.BatchNorm1d(kernel_layer[0], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(24, self.model_shape[1], bias=True)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x0 = unsqueeze(x, dim=1)
        encoded0 = self.encoder(x0)
        encoded, indices = self.pool(encoded0)
        decoded0 = self.un_pool(encoded, indices)
        decoded = self.decoder(decoded0)
        return self.flatten(encoded), self.flatten(decoded)


# Anpassungen an fcnn_layer[0] und fcnn_out notwendig, wenn CNN-Kernel geändert wird
class cnn_ae_v3(nn.Module):
    """Class of a convolutional autoencoder for feature extraction"""
    def __init__(self, input_size=32, output_size=6):
        super().__init__()
        self.out_modelname = 'cnn_ae_v3'
        self.out_modeltyp = 'Autoencoder'
        self.model_embedded = False
        self.model_shape = (1, input_size)
        do_bias_train = True
        kernel_layer = [1, 40, 22, 8]
        kernel_size = [4, 3, 3]
        kernel_stride = [1, 2, 2]
        kernel_padding = [0, 0, 0]
        kernel_out = [0, 0, 0]
        fcnn_layer = [24, 14, output_size]
        fcnn_out = 199

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm1d(kernel_layer[3], affine=do_bias_train),
            nn.Tanh(),
            nn.Flatten()
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(fcnn_layer[0], fcnn_layer[1], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_layer[1], fcnn_layer[2], bias=do_bias_train)
        )
        self.flatten = nn.Flatten(start_dim=1)
        # Decoder setup
        self.decoder_linear = nn.Sequential(
            nn.BatchNorm1d(fcnn_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_layer[2], fcnn_layer[1], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_layer[1], fcnn_layer[0], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[0], affine=do_bias_train),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, kernel_layer[2], kernel_size[2], stride=kernel_stride[2],
                               padding=kernel_padding[2], output_padding=kernel_out[2]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[2], kernel_layer[1], kernel_size[1], stride=kernel_stride[1],
                               padding=kernel_padding[1], output_padding=kernel_out[1]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[1], kernel_layer[0], kernel_size[0], stride=kernel_stride[0],
                               padding=kernel_padding[0], output_padding=kernel_out[0]),
            nn.BatchNorm1d(kernel_layer[0], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_out, self.model_shape[1], bias=do_bias_train)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x0 = unsqueeze(x, dim=1)
        x0 = self.encoder(x0)
        encoded = self.encoder_linear(x0)
        decoded0 = self.decoder_linear(encoded)
        decoded0 = unsqueeze(decoded0, dim=1)
        decoded = self.decoder(decoded0)

        return encoded, self.flatten(decoded)


Recommended_Config_PytorchSettings = Config_PyTorch(
    model=dnn_ae_v1(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=40,
    batch_size=256,
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
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
