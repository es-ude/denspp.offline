from torch import nn, Tensor
from package.dnn.pytorch_handler import __model_settings_common, ModelRegistry


models_available = ModelRegistry()


@models_available.register
class dnn_ae_rgc_fzj_v1(__model_settings_common):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=40, output_size=6):
        super().__init__('Autoencoder')
        self.model_shape = (1, input_size)
        self.model_embedded = False
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        iohiddenlayer = [input_size, 24, output_size]

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


@models_available.register
class dnn_ae_rgc_fzj_v2(__model_settings_common):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=40, output_size=4):
        super().__init__('Autoencoder')
        self.model_shape = (1, input_size)
        self.model_embedded = False
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        iohiddenlayer = [input_size, 28, 14, output_size]

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[3], affine=do_train_batch)
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


@models_available.register
class dnn_ae_rgc_tdb_v1(__model_settings_common):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=32, output_size=3):
        super().__init__('Autoencoder')
        self.model_shape = (1, input_size)
        self.model_embedded = False
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        iohiddenlayer = [input_size, 20, output_size]

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1],
                      bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1],
                           affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2],
                      bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2],
                           affine=do_train_batch)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1],
                      bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1],
                           affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0],
                      bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)
