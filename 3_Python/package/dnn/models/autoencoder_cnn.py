from torch import nn, Tensor, unsqueeze, argmax
from package.dnn.pytorch_handler import __model_settings_common, ModelRegistry


models_available = ModelRegistry()


@models_available.register
class cnn_ae_v1(__model_settings_common):
    def __init__(self, input_size=32, output_size=5):
        """Class of a convolutional autoencoder for feature extraction"""
        super().__init__('Autoencoder')
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


@models_available.register
class cnn_ae_v2(__model_settings_common):
    def __init__(self, input_size=32, output_size=5):
        """Class of a convolutional autoencoder for feature extraction"""
        super().__init__('Autoencoder')
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


@models_available.register
class cnn_ae_v3(__model_settings_common):
    def __init__(self, input_size=32, output_size=6):
        """Class of a convolutional autoencoder for feature extraction"""
        super().__init__('Autoencoder')
        self.model_embedded = False
        self.model_shape = (1, 32)
        do_bias_train = True
        kernel_layer = [1, 40, 22, 8]
        kernel_size = [4, 3, 3]
        kernel_stride = [1, 2, 2]
        kernel_padding = [0, 0, 0]
        kernel_out = [0, 0, 0]
        fcnn_layer = [48, 20, output_size]
        fcnn_out = 198

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


@models_available.register
class cnn_ae_v4(__model_settings_common):
    def __init__(self, input_size=32, output_size=8):
        """Class of a convolutional autoencoder for feature extraction"""
        super().__init__('Autoencoder')
        self.model_embedded = False
        self.model_shape = (1, 32)
        do_bias_train = True
        kernel_layer = [1, 42, 22, output_size]
        kernel_size = [3, 3, 3]
        kernel_stride = [1, 2, 2]
        kernel_padding = [0, 0, 0]
        kernel_pool_size = [2, 2, 2]
        kernel_pool_stride = [1, 2, 2]
        fcnn_layer = [output_size, 12, 16, 22, 26, 32]

        # Encoder setup
        self.encoder = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_bias_train),
            nn.Tanh(),
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
        x0 = unsqueeze(x, dim=1)
        encoded = self.encoder(x0)
        decoded = self.decoder(encoded)

        return encoded, decoded
