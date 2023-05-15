import torch
import torch.nn as nn

class sda_bae_v1(nn.Module):
    """Class to identify spike type from NEO operator"""
    def __init__(self):
        super().__init__()

        task = "Classification"
        iohiddenlayer = [32, 20, 12, 4]
        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[0]),
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3]),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)