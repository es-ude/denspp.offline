from torch import nn, Tensor, unsqueeze, argmax, cuda
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset


class cnn_rgc_ae_v1(nn.Module):
    """Class of a convolutional autoencoder for feature extraction"""
    def __init__(self, input_size=32, output_size=8):
        super().__init__()
        self.out_modelname = 'cnn_ae_v4'
        self.out_modeltyp = 'Autoencoder'
        self.model_embedded = False
        self.model_shape = (1, input_size)
        do_bias_train = True
        kernel_layer = [1, 42, 22, output_size]
        kernel_size = [3, 3, 3]
        kernel_stride = [1, 2, 2]
        kernel_padding = [0, 0, 0]
        kernel_pool_size = [2, 2, 2]
        kernel_pool_stride = [1, 2, 2]
        fcnn_layer = [output_size, 12, 16, 22, 26, input_size]

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.ReLU(),
            nn.AvgPool1d(kernel_pool_size[0], kernel_pool_stride[0]),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_bias_train),
            nn.ReLU(),
            nn.AvgPool1d(kernel_pool_size[1], kernel_pool_stride[1]),
            nn.Conv1d(kernel_layer[2], kernel_layer[3], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2]),
            nn.BatchNorm1d(kernel_layer[3], affine=do_bias_train),
            nn.ReLU(),
            nn.AvgPool1d(kernel_pool_size[2], kernel_pool_stride[2]),
            nn.Flatten()
        )
        # Decoder setup
        self.decoder = nn.Sequential(
            nn.Linear(fcnn_layer[0], fcnn_layer[1], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[1], affine=do_bias_train),
            nn.ReLU(),
            nn.Linear(fcnn_layer[1], fcnn_layer[2], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[2], affine=do_bias_train),
            nn.ReLU(),
            nn.Linear(fcnn_layer[2], fcnn_layer[3], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[3], affine=do_bias_train),
            nn.ReLU(),
            nn.Linear(fcnn_layer[3], fcnn_layer[4], bias=do_bias_train),
            nn.BatchNorm1d(fcnn_layer[4], affine=do_bias_train),
            nn.ReLU(),
            nn.Linear(fcnn_layer[4], fcnn_layer[5], bias=do_bias_train)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x0 = unsqueeze(x, dim=1)
        encoded = self.encoder(x0)
        decoded = self.decoder(encoded)
        return encoded, decoded


class rgc_ae_cl_v2(nn.Module):
    """Classification model"""
    def __init__(self, input_size=6, output_size=4):
        super().__init__()
        self.out_modelname = 'rgc_class_v2'
        self.out_modeltyp = 'Classification'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 64, 72, 58, 36, 24, output_size]
        lin_drop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(lin_size[0], lin_size[1], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[0]),
            nn.Linear(lin_size[1], lin_size[2], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[1]),
            nn.Linear(lin_size[2], lin_size[3], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[2]),
            nn.Linear(lin_size[3], lin_size[4], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[4], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[3]),
            nn.Linear(lin_size[4], lin_size[5], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[5], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[4]),
            nn.Linear(lin_size[5], lin_size[6], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[6], affine=do_train_bias),
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)


class rgc_ae_cl_v1(nn.Module):
    """Classification model of autoencoder output"""
    def __init__(self, input_size=6, output_size=5):
        super().__init__()
        self.out_modelname = 'ae_class_v1'
        self.out_modeltyp = 'Classification'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 16, 12, output_size]
        lin_drop = [0.0, 0.0]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[0]),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[1]),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)


