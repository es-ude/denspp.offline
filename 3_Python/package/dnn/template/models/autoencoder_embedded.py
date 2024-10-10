import torch
import torch.nn as nn
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
# Check errors
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import BatchNormedLinear, HardTanh
from package.dnn.pytorch_handler import __model_settings_common, ModelRegistry


models_available = ModelRegistry()


@models_available.register
class dnn_ae_v2(__model_settings_common):
    def __init__(self, input_size=32, output_size=3):
        super().__init__('Autoencoder')
        self.model_shape = (1, input_size)
        self.model_embedded = True
        bits_total = 12
        bits_frac = 9
        iohiddenlayer = [input_size, 20, output_size]
        do_train_bias = True

        # --- Encoder Path
        self.encoder = Sequential(
            BatchNormedLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1],
                              total_bits=bits_total, frac_bits=bits_frac,
                              bias=do_train_bias),
            HardTanh(total_bits=bits_total, frac_bits=bits_frac),
            BatchNormedLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2],
                              total_bits=bits_total, frac_bits=bits_frac,
                              bias=do_train_bias)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[2]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2],
                      out_features=iohiddenlayer[1]),
            nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1],
                      out_features=iohiddenlayer[0])
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)
