from torch import nn, Tensor, argmax
from package.dnn.pytorch_control import Config_PyTorch


class dnn_rgc_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=40, output_size=5):
        super().__init__()
        self.out_modelname = 'rgc_class_v1'
        self.out_modeltyp = 'Classification'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 45, 32, 28, 16, output_size]
        rate_drop = [0.1, 0.1, 0.15, 0.05]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.Tanh(),
            nn.Dropout(rate_drop[0]),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.Tanh(),
            nn.Dropout(rate_drop[1]),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(rate_drop[2]),
            nn.Linear(lin_size[3], lin_size[4]),
            nn.BatchNorm1d(lin_size[4], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(rate_drop[3]),
            nn.Linear(lin_size[4], lin_size[5]),
            nn.Softmax()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)


class dnn_rgc_v2(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self, input_size=40, output_size=5):
        super().__init__()
        self.out_modelname = 'rgc_class_v2'
        self.out_modeltyp = 'Classification'
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 45, 32, 28, 16, output_size]
        rate_drop = [0.01, 0.01, 0.01, 0.01]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(rate_drop[0]),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(rate_drop[1]),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(rate_drop[2]),
            nn.Linear(lin_size[3], lin_size[4]),
            nn.BatchNorm1d(lin_size[4], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(rate_drop[3]),
            nn.Linear(lin_size[4], lin_size[5]),
            nn.Softmax()
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)


Recommended_Config_PytorchSettings = Config_PyTorch(
    model=dnn_rgc_v1(),
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
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