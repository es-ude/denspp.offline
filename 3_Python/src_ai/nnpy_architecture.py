import torch
import torch.nn as nn
import torchvision.models as models

class dnn_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mname = "dnn_dae_v2"
        self.optimizer_param = None

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=40, out_features=28),
            nn.Tanh(),
            nn.Linear(in_features=28, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=8),
            nn.Tanh()
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Linear(in_features=8, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=28),
            nn.Tanh(),
            nn.Linear(in_features=28, out_features=40),
            #nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)