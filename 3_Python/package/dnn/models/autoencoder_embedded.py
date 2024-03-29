import torch
import torch.nn as nn
from package.dnn.pytorch.handler import Config_PyTorch, Config_Dataset
# Using Elastic-AI.creator version: 0.57.1
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import BatchNormedLinear, HardTanh


class dnn_ae_v2(nn.Module):
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.out_modelname = 'dnn_dae_embedded_v2'
        self.out_modeltyp = 'Autoencoder'
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


Recommended_Config_PytorchSettings = Config_PyTorch(
    model=dnn_ae_v2(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=40,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.25
)

Recommended_Config_DatasetSettings = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='',
    data_normalization_method='',
    data_normalization_setting='',
    # --- Dataset Preparation
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=0,
    data_exclude_cluster=[],
    data_sel_pos=[]
)
