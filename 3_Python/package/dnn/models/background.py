from torch import nn, Tensor

class sda_bae_v1(nn.Module):
    """Class to identify spike type from NEO operator"""
    def __init__(self):
        super().__init__()
        task = "Classification"
        iohiddenlayer = [32, 20, 12, 4]
        use_bias = False

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[0], bias=use_bias),
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=use_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=use_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)