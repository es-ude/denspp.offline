import torch
import torch.nn as nn

class dnn_dae_v1(nn.Module):
    def __init__(self):
        super().__init__()

        iohiddenlayer = [32, 20, 3]
        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[0]),
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2])
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0])
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

class dnn_dae_v2(nn.Module):
    def __init__(self):
        super().__init__()

        iohiddenlayer = [32, 24, 10]
        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(),
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2]),
            nn.Tanh()
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1]),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0])
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

class cnn_dae_v1(nn.Module):
    def __init__(self):
        super().__init__()

        iohiddenlayer = [40, 20, 8]
        kernelC = [3, 3]
        kernelM = [2, 2]
        # Encoder setup
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[0]),
            nn.Conv1d(in_channels=iohiddenlayer[0], out_channels=iohiddenlayer[1], kernel_size=kernelC[0], padding='same'),
            nn.MaxPool1d(kernel_size=kernelM[0], padding='same'),
            nn.Tanh(), nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Conv1d(in_channels=iohiddenlayer[1], out_channels=iohiddenlayer[2], kernel_size=kernelC[1], padding='same'),
            nn.MaxPool1d(kernel_size=kernelM[1], padding='same'),
            nn.Tanh(), nn.BatchNorm1d(num_features=iohiddenlayer[2])
        )

        self.flatten = nn.Flatten(start_dim=1)

        # Decoder setup
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_features=iohiddenlayer[2]),
            nn.Conv1d(in_channels=iohiddenlayer[2], out_channels=iohiddenlayer[1], kernel_size=kernelC[0], padding='same'),
            nn.Upsample(scale_factor=kernelM[1]),
            nn.Tanh(), nn.BatchNorm1d(num_features=iohiddenlayer[1]),
            nn.Conv1d(in_channels=iohiddenlayer[1], out_channels=iohiddenlayer[0], kernel_size=kernelC[1], padding='same'),
            nn.Upsample(scale_factor=kernelM[0]),
            #decoded0 = nntf.layers.Conv1D(1, 3, padding='same')(decoded)
            #nn.Tanh(), nn.BatchNorm1d(num_features=iohiddenlayer[0])
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        encoded = self.flatten(self.encoder(x))
        decoded = self.flatten(self.encoded(encoded))
        return encoded, decoded