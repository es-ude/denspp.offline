import torch
import torch.nn as nn

from elasticai.creator.nn.vhdl.linear import FPLinear

class dnn_dae_v1(nn.Module):
    def __init__(self):
        super().__init__()
        bits_total = 6
        bits_frac = 4
        iohiddenlayer = [32, 20, 3]

        # --- Encoder Path
        self.encoder = nn.Sequential(
            FPLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], total_bits=bits_total, frac_bits=bits_frac, bias=True),
            nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Tanh(),
            FPLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], total_bits=bits_total, frac_bits=bits_frac, bias=True)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[2]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1]),
            nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0]),
            nn.BatchNorm1d(num_features=iohiddenlayer[0])
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)