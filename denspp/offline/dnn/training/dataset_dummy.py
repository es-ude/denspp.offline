import numpy as np
from torch import nn, Tensor, argmax, flatten
from denspp.offline.dnn import DatasetFromFile


def generate_dummy_dataset(num_samples: int, num_window: int) -> DatasetFromFile:
    sample = np.random.randn(num_samples, num_window)
    label = np.zeros(shape=(num_samples, ))
    xpos = np.argwhere(np.mean(sample, axis=1) > 0).flatten()
    xneg = np.argwhere(np.mean(sample, axis=1) <= 0).flatten()
    label[xpos] = 1
    label[xneg] = 0
    return DatasetFromFile(
        data=sample,
        label=label,
        dict=['zero', 'one'],
        mean=np.zeros(shape=(2, num_window))
    )


class dummy_mlp_cl_v0(nn.Module):
    def __init__(self, input_size: int=10, output_size: int=2):
        super().__init__()
        self.model_shape = (1, input_size)
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [input_size, 40, output_size]

        # --- Model Deployment
        self.model = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.model.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[idx-1], out_features=layer_size, bias=do_train_bias))
            self.model.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network)-1:
                self.model.add_module(f"act_{idx:02d}", nn.ReLU())
            else:
                pass

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, dim=1)


class dummy_mlp_ae_v0(nn.Module):
    def __init__(self, input_size: int=280, output_size: int=4):
        super().__init__()
        self.model_shape = (1, input_size)
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [input_size, 120, 36, output_size]

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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
