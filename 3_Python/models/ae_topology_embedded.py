import torch
import torch.nn as nn

from elasticai.creator.nn import FPLinear, FPHardTanh, FPBatchNormedLinear, Sequential

class dnn_dae_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_dae_embedded_v1'
        self.out_modeltyp = 'dae'
        bits_total = 12
        bits_frac = 9
        iohiddenlayer = [32, 20, 3]
        use_bias = False

        # --- Encoder Path
        self.encoder = nn.Sequential(
            FPLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], total_bits=bits_total, frac_bits=bits_frac, bias=use_bias),
            # nn.BatchNorm1d(num_features=iohiddenlayer[1], bias=use_bias),
            FPHardTanh(total_bits=bits_total, frac_bits=bits_frac),
            FPLinear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], total_bits=bits_total, frac_bits=bits_frac, bias=use_bias)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(num_features=iohiddenlayer[2], bias=use_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=use_bias),
            # nn.BatchNorm1d(num_features=iohiddenlayer[1], bias=use_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=use_bias)
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

class dnn_dae_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_modelname = 'dnn_dae_embedded_v2'
        self.out_modeltyp = 'dae'
        bits_total = 12
        bits_frac = 9
        iohiddenlayer = [32, 20, 3]

        # --- Encoder Path
        self.encoder = Sequential(
            FPBatchNormedLinear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], total_bits=bits_total, frac_bits=bits_frac, bias=True),
            FPHardTanh(total_bits=bits_total, frac_bits=bits_frac),
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