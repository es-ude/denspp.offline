import torch
from torch import nn, Tensor, argmax
from package.dnn.pytorch_handler import __model_settings_common, ModelRegistry


models_available = ModelRegistry()


@models_available.register
class mnist_mlp_cl_v1(__model_settings_common):
    """Class of a classifier with Dense-Layer for feature extraction"""
    def __init__(self):
        super().__init__('Classifier')
        self.model_shape = (1, 28, 28)
        self.model_embedded = False
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [784, 40, 10]

        # --- Model Deployment
        self.model = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.model.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[idx-1], out_features=layer_size, bias=do_train_bias))
            self.model.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network)-1:
                self.model.add_module(f"act_{idx:02d}", nn.ReLU())
            else:
                # self.model.add_module(f"soft", nn.Softmax(dim=1))
                pass

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = torch.flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


@models_available.register
class mnist_mlp_ae_v1(__model_settings_common):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self):
        super().__init__('Autoencoder')
        self.model_shape = (1, 28, 28)
        self.model_embedded = False
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [784, 40, 10]

        # --- Model Deployment: Encoder
        self.encoder = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.encoder.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[idx - 1], out_features=layer_size, bias=do_train_bias))
            self.encoder.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network) - 1:
                self.encoder.add_module(f"act_{idx:02d}", nn.ReLU())

        # --- Model Deployment: Decoder
        self.decoder = nn.Sequential()
        for idx, layer_size in enumerate(reversed(config_network[:-1]), start=1):
            if idx == 1:
                self.decoder.add_module(f"act_dec_{idx:02d}", nn.ReLU())
            self.decoder.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[-idx], out_features=layer_size, bias=do_train_bias))
            if not idx == len(config_network) - 1:
                self.decoder.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
                self.decoder.add_module(f"act_{idx:02d}", nn.ReLU())

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = torch.flatten(x, start_dim=1)
        encoded = self.encoder(x)
        return encoded, torch.reshape(self.decoder(encoded), (x.shape[0], 28, 28))
