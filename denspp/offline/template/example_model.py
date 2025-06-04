from torch import nn, Tensor, argmax, flatten, reshape


# Comment #1: Please end the class name with '_v' and a number. The used ModelRegistry searches in all files for this pattern
# Comment #2: If you are using a classification model/architecture, please provide a forward func with output of the probability and the label id.
class mnist_mlp_cl_example_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)
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
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


# Comment #2: If you are using an autoencoder model/architecture, please provide a forward func with output of the features (encoder output) and the reconstructed signal (decoder output).
class mnist_mlp_ae_example_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)
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
        x = flatten(x, start_dim=1)
        encoded = self.encoder(x)
        return encoded, reshape(self.decoder(encoded), (x.shape[0], 28, 28))
