import torch
import torch.nn as nn


class dnn_ae_v1(nn.Module):
    """Class for using an autoencoder with Dense-Layer"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_ae_v1'
        self.out_modeltyp = 'ae'
        iohiddenlayer = [32, 20, 14, 3]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[0], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[3], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class dnn_dae_v1(nn.Module):
    """Class for using a denoising autoencoder with Dense-Layer"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_dae_v1'
        self.out_modeltyp = 'dae'
        iohiddenlayer = [32, 20, 3]
        do_train_bias = False
        do_train_batch = False

        # --- Encoder Path
        self.encoder = nn.Sequential(
            #nn.BatchNorm1d(num_features=iohiddenlayer[0], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.Tanh(),
            #nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class cnn_ae_v1(nn.Module):
    """Class for using a convolutional autoencoder"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'cnn_ae_v1'
        self.out_modeltyp = 'ae'
        iohiddenlayer = [32, 20, 3]
        kernelC = [3, 3]
        pool_stride = [2, 2]

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[0]),
            nn.Conv1d(in_channels=iohiddenlayer[0], out_channels=iohiddenlayer[1],
                      kernel_size=kernelC[0], stride=1, padding='same'),
            nn.MaxPool1d(kernel_size=pool_stride[0]),
            nn.Tanh(), nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Conv1d(in_channels=iohiddenlayer[1], out_channels=iohiddenlayer[2],
                      kernel_size=kernelC[1], stride=1, padding='same'),
            nn.MaxPool1d(kernel_size=pool_stride[1]),
            nn.Tanh(), nn.BatchNorm1d(num_features=iohiddenlayer[2])
        )

        self.flatten = nn.Flatten(start_dim=1)

        # Decoder setup
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[2]),
            nn.Conv1d(in_channels=iohiddenlayer[2], out_channels=iohiddenlayer[1],
                      kernel_size=kernelC[0], stride=1, padding='same'),
            nn.Upsample(scale_factor=pool_stride[1]),
            nn.Tanh(), nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Conv1d(in_channels=iohiddenlayer[1], out_channels=iohiddenlayer[0],
                      kernel_size=kernelC[1], stride=1, padding='same'),
            nn.Upsample(scale_factor=pool_stride[0])
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.flatten(self.encoder(x))
        decoded = self.flatten(self.encoded(encoded))
        return encoded, decoded
