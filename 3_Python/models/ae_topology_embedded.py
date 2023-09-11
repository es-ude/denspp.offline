import torch
import torch.nn as nn
# Using Elastic-AI.creator version: 0.57.1
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, BatchNormedLinear, HardTanh


class dnn_dae_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_dae_embedded_v2'
        self.out_modeltyp = 'dae'
        self.model_shape = (1, 32)
        self.out_embedded = True
        bits_total = 12
        bits_frac = 9
        iohiddenlayer = [self.model_shape[1], 20, 3]
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
