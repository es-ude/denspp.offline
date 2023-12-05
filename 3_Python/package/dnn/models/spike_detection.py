from torch import nn, Tensor, unsqueeze, argmax
from package.dnn.pytorch_control import Config_PyTorch


class dnn_sda_v1(nn.Module):
    """Class of a dense-layer based spike detection classifier"""
    def __init__(self, input_size=9, output_size=2):
        super().__init__()
        self.out_modelname = 'dnn_sda_v1'
        self.out_modeltyp = 'Classification'
        self.model_embedded = False
        self.model_shape = (1, input_size)
        lin_size = [input_size, 6, 4, output_size]
        do_train_bias = True
        self.out_class = ['Spike', 'Non-Spike']

        self.detector = nn.Sequential(
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.ReLU(),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.ReLU(),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.Softmax(dim=0)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        xdist = self.detector(x)
        return xdist, argmax(xdist, dim=1)


class cnn_sda_v1(nn.Module):
    """Class of a convolutional spike detection classifier"""
    def __init__(self, input_size=9, output_size=2):
        super().__init__()
        self.out_modelname = 'cnn_sda_v1'
        self.out_modeltyp = 'Classification'
        self.model_embedded = False
        self.model_shape = (1, input_size)
        kernel_layer = [1, 4, 3]
        kernel_size = [3, 3]
        kernel_stride = [2, 1]
        kernel_padding = [0, 0, 0]
        pool_size = [2]
        lin_size = [3, output_size]
        do_train_bias = True
        self.out_class = ['Spike', 'Non-Spike']

        self.detector = nn.Sequential(
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0]),
            nn.BatchNorm1d(kernel_layer[1], affine=do_train_bias),
            nn.Tanh(),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1]),
            nn.BatchNorm1d(kernel_layer[2], affine=do_train_bias),
            nn.Tanh(),
            nn.MaxPool1d(pool_size[0]),
            nn.Flatten(start_dim=1),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.Softmax(dim=0)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        xin = unsqueeze(x, dim=1)
        xds = self.detector(xin)
        return xds, argmax(xds, dim=1)


Recommended_Config_PytorchSettings = Config_PyTorch(
    model=dnn_sda_v1(),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=40,
    batch_size=256,
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=0,
    # --- Dataset Preparation
    data_exclude_cluster=[],
    data_sel_pos=[]
)