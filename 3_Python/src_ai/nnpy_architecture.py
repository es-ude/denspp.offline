import torch
import torch.nn as nnpy
import torchvision.models as models

class dnn_autoencoder(nnpy.Module):
    def __init__(self):
        super().__init__()
        self.mname = "dnn_dae_v2"
        self.optimizer_param = None

        # --- Encoder Path
        self.encoder = nnpy.Sequential(
            nnpy.Linear(in_features=40, out_features=28, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=28, out_features=12, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=12, out_features=8, bias=False).double(),
            nnpy.Tanh()
        )
        # --- Decoder Path
        self.decoder = nnpy.Sequential(
            nnpy.Linear(in_features=8, out_features=12, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=12, out_features=28, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=28, out_features=40, bias=False).double(),
            nnpy.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)