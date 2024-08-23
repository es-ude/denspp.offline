import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision
from torch import nn, Tensor, unsqueeze, argmax


class cnn_ae_v1_2d(nn.Module):
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
        kernel_size = [(4, 3), (3, 3), (3, 3)]
        kernel_stride = [(2, 2), (2, 2), (2, 2)]
        kernel_padding = [(0, 0), (0, 0), (0, 0)]
        kernel_out = [(0, 0), (0, 0), (0, 0)]
        pool_size = [(2, 2), (2, 2)]
        pool_stride = [(2, 2), (2, 2)]

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm2d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv2d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm2d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv2d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm2d(kernel_layer[3], affine=do_bias_train)
        )
        self.pool = nn.MaxPool2d(pool_size[0], stride=pool_stride[0], return_indices=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.un_pool = nn.MaxUnpool2d(pool_size[1], stride=pool_stride[1])
        # Decoder setup
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.ConvTranspose2d(kernel_layer[3], kernel_layer[2], kernel_size[2], stride=kernel_stride[2],
                               padding=kernel_padding[2], output_padding=kernel_out[2]),
            nn.BatchNorm2d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose2d(kernel_layer[2], kernel_layer[1], kernel_size[1], stride=kernel_stride[1],
                               padding=kernel_padding[1], output_padding=kernel_out[1]),
            nn.BatchNorm2d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose2d(kernel_layer[1], kernel_layer[0], kernel_size[0], stride=kernel_stride[0],
                               padding=kernel_padding[0], output_padding=kernel_out[0]),
            nn.BatchNorm2d(kernel_layer[0], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(24, self.model_shape[1], bias=True)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded0 = self.encoder(x)
        encoded, indices = self.pool(encoded0)
        decoded0 = self.un_pool(encoded, indices)
        decoded = self.decoder(decoded0)
        return self.flatten(encoded), self.flatten(decoded)




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
        kernel_size = [(4, 3), (3, 3), (3, 3)]
        kernel_stride = [(1, 1), (2, 2), (2, 2)]
        kernel_padding = [(0, 0), (0, 0), (0, 0)]
        kernel_out = [(0, 0), (0, 0), (0, 0)]
        fcnn_layer = [48, 20, output_size]
        fcnn_out = 198

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm2d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv2d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm2d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.Conv2d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm2d(kernel_layer[3], affine=do_bias_train),
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
            nn.ConvTranspose2d(1, kernel_layer[2], kernel_size[2], stride=kernel_stride[2],
                               padding=kernel_padding[2], output_padding=kernel_out[2]),
            nn.BatchNorm2d(kernel_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose2d(kernel_layer[2], kernel_layer[1], kernel_size[1], stride=kernel_stride[1],
                               padding=kernel_padding[1], output_padding=kernel_out[1]),
            nn.BatchNorm2d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.ConvTranspose2d(kernel_layer[1], kernel_layer[0], kernel_size[0], stride=kernel_stride[0],
                               padding=kernel_padding[0], output_padding=kernel_out[0]),
            nn.BatchNorm2d(kernel_layer[0], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_out, self.model_shape[1], bias=do_bias_train)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x0 = self.encoder(x)
        encoded = self.encoder_linear(x0)
        decoded0 = self.decoder_linear(encoded)
        decoded = self.decoder(decoded0)

        return encoded, self.flatten(decoded)




class cnn_ae_v4(nn.Module):
    """Class of a convolutional autoencoder for feature extraction"""
    def __init__(self, input_size=32, output_size=8):
        super().__init__()
        self.out_modelname = 'cnn_ae_v4'
        self.out_modeltyp = 'Autoencoder'
        self.model_embedded = False
        self.model_shape = (1, input_size)
        do_bias_train = True
        kernel_layer = [1, 42, 22, output_size]
        kernel_size = [(3, 3), (3, 3), (3, 3)]
        kernel_stride = [(1, 1), (2, 2), (2, 2)]
        kernel_padding = [(0, 0), (0, 0), (0, 0)]
        kernel_pool_size = [(2, 2), (2, 2), (2, 2)]
        kernel_pool_stride = [(1, 1), (2, 2), (2, 2)]
        fcnn_layer = [output_size, 12, 16, 22, 26, input_size]

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv2d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm2d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.AvgPool2d(kernel_pool_size[0], kernel_pool_stride[0]),
            nn.Conv2d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm2d(kernel_layer[2], affine=do_bias_train),
            nn.ReLU(),
            nn.AvgPool2d(kernel_pool_size[1], kernel_pool_stride[1]),
            nn.Conv2d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm2d(kernel_layer[3], affine=do_bias_train),
            nn.ReLU(),
            nn.AvgPool2d(kernel_pool_size[2], kernel_pool_stride[2]),
            nn.Flatten()
        )
        # Decoder setup
        self.decoder = nn.Sequential(
            nn.Linear(fcnn_layer[0], fcnn_layer[1], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[1], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_layer[1], fcnn_layer[2], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[2], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_layer[2], fcnn_layer[3], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[3], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_layer[3], fcnn_layer[4], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[4], affine=do_bias_train),
            nn.Tanh(),
            nn.Linear(fcnn_layer[4], fcnn_layer[5], bias=do_bias_train)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

import torch.nn as nn

class classifier_ae_v1(nn.Module):
    """Classification model of autoencoder output"""
    def __init__(self, input_channels=6, output_size=5):
        super().__init__()
        self.out_modelname = 'ae_class_v1'
        self.out_modeltyp = 'Classification'
        self.model_shape = (input_channels, 1)
        self.model_embedded = False
        conv_channels = [input_channels, 16, 12]
        kernel_size = 3
        pool_size = 2
        lin_size = [12 * (input_channels // pool_size**2), output_size]
        lin_drop = [0.0, 0.0]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size),
            nn.BatchNorm2d(conv_channels[1], affine=do_train_bias),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size),
            nn.BatchNorm2d(conv_channels[2], affine=do_train_bias),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Flatten(),
            nn.Dropout(lin_drop[0]),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[1]),
            nn.Linear(lin_size[1], output_size),
            nn.BatchNorm1d(output_size, affine=do_train_bias),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(x)


import torch.nn as nn

class Config_PyTorch:
    def __init__(self):
        self.model = cnn_ae_v1_2d()
        self.loss = 'MSE'
        self.loss_fn = nn.MSELoss()
        self.optimizer = 'Adam'
        self.num_kfold = 1
        self.num_epochs = 40
        self.batch_size = 256
        self.data_do_shuffle = True
        self.data_split_ratio = 0.25

class Config_Dataset:
    def __init__(self):
        self.data_path = '../2_Data/00_Merged_Datasets'
        self.data_file_name = '2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat'
        self.data_do_augmentation = False
        self.data_num_augmentation = 0
        self.data_do_normalization = False
        self.data_do_addnoise_cluster = False
        self.data_do_reduce_samples_per_cluster = False
        self.data_num_samples_per_cluster = 0
        self.data_exclude_cluster = []
        self.data_sel_pos = []