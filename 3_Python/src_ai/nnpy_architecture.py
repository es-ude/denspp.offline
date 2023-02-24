import torch
import torch.nn as nnpy
import torchvision.models as models

class nn_autoencoder (nnpy.Module):
    def __init__(self, io_size: int):
        super().__init__()

        self.encoder = nnpy.Sequential(
            # --- Encoder Path
            nnpy.Linear(in_features=io_size, out_features=32, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=32, out_features=20, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=20, out_features=8, bias=False).double(),
            nnpy.Tanh()
        )

        self.decoder = nnpy.Sequential(
            # --- Decoder Path
            nnpy.Linear(in_features=8, out_features=20, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=20, out_features=32, bias=False).double(),
            nnpy.Tanh(),
            nnpy.Linear(in_features=32, out_features=io_size, bias=False).double()
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded
