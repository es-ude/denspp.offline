from torch import nn, Tensor
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset


class dnn_ae_rgc_fzj_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=40, output_size=6):
        super().__init__()
        self.out_modelname = 'dnn_rgc_fzj_ae_v1'
        self.out_modeltyp = 'Autoencoder'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 24, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class dnn_ae_rgc_fzj_v2(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=40, output_size=4):
        super().__init__()
        self.out_modelname = 'dnn_rgc_fzj_ae_v2'
        self.out_modeltyp = 'Autoencoder'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 28, 14, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[3], affine=do_train_batch)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[3], out_features=iohiddenlayer[2], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1], bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1], affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0], bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class dnn_ae_rgc_tdb_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=32, output_size=3):
        super().__init__()
        self.out_modelname = 'dnn_rgc_tdb_ae_v1'
        self.out_modeltyp = 'Autoencoder'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        iohiddenlayer = [input_size, 20, output_size]
        do_train_bias = True
        do_train_batch = True

        # --- Encoder Path
        self.encoder = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1],
                      bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1],
                           affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2],
                      bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[2],
                           affine=do_train_batch)
        )
        # --- Decoder Path
        self.decoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[1],
                      bias=do_train_bias),
            nn.BatchNorm1d(num_features=iohiddenlayer[1],
                           affine=do_train_batch),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[0],
                      bias=do_train_bias)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


Recommended_Config_PytorchSettings = Config_PyTorch(
    model=dnn_ae_rgc_tdb_v1(),
    loss_fn=nn.MSELoss(),
    loss='MSE',
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
