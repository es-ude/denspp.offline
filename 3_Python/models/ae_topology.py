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


class dnn_ae_v2(nn.Module):
    """Class for using an autoencoder with Dense-Layer"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_ae_v2'
        self.out_modeltyp = 'dae'
        iohiddenlayer = [32, 20, 3]
        do_train_bias = False
        do_train_batch = False

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
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
        kernel_layer = [1, 3, 8, 4]
        kernel_size = [2, 2, 2]
        kernel_stride = [1, 2, 1]
        kernel_padding = [0, 1, 0]
        kernel_out = [0, 1, 0]
        pool_size = [2, 2]
        pool_stride = [2, 2]

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.Tanh(),
            nn.BatchNorm1d(kernel_layer[1]),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            #nn.Tanh(),
            #nn.MaxPool1d(pool_size[0], stride=pool_stride[0], return_indices=True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        # Decoder setup
        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(pool_size[0], stride=pool_stride[0]),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[3], kernel_layer[2], kernel_size[2], stride=kernel_stride[2],
                               padding=kernel_padding[2], output_padding=kernel_out[2]),
            nn.Tanh(),
            nn.BatchNorm1d(kernel_layer[2]),
            nn.ConvTranspose1d(kernel_layer[2], kernel_layer[1], kernel_size[1], stride=kernel_stride[1],
                               padding=kernel_padding[1], output_padding=kernel_out[1]),
            nn.Tanh(),
            nn.ConvTranspose1d(kernel_layer[1], kernel_layer[0], kernel_size[0], stride=kernel_stride[0],
                               padding=kernel_padding[0], output_padding=kernel_out[0])
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        x0 = torch.unsqueeze(x, dim=1)
        encoded = self.encoder(x0)
        decoded = self.decoder(encoded)
        return self.flatten(encoded), self.flatten(decoded)
