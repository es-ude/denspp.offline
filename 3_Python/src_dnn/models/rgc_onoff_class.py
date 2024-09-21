from torch import nn, Tensor, argmax, unsqueeze
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
from package.dnn.pytorch_handler import __model_settings_common


class dnn_rgc_v1(__model_settings_common):
    """Classification model"""
    def __init__(self, input_size=40, output_size=4):
        super().__init__('Classifier')
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 45, 32, 28, 16, output_size]
        lin_drop = [0.1, 0.1, 0.15, 0.05]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(lin_size[0], lin_size[1]),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.Tanh(),
            nn.Dropout(lin_drop[0]),
            nn.Linear(lin_size[1], lin_size[2]),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.Tanh(),
            nn.Dropout(lin_drop[1]),
            nn.Linear(lin_size[2], lin_size[3]),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[2]),
            nn.Linear(lin_size[3], lin_size[4]),
            nn.BatchNorm1d(lin_size[4], affine=do_train_bias),
            nn.ReLU(),
            nn.Dropout(lin_drop[3]),
            nn.Linear(lin_size[4], lin_size[5]),
            # nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)


class dnn_rgc_v2(__model_settings_common):
    """Classification model"""
    def __init__(self, input_size=40, output_size=4):
        super().__init__('Classifier')
        self.model_shape = (1, input_size)
        self.model_embedded = False
        lin_size = [input_size, 64, 72, 58, 36, 24, output_size]
        lin_drop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        do_train_bias = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(lin_size[0], lin_size[1], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[0]),
            nn.Linear(lin_size[1], lin_size[2], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[1]),
            nn.Linear(lin_size[2], lin_size[3], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[2]),
            nn.Linear(lin_size[3], lin_size[4], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[4], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[3]),
            nn.Linear(lin_size[4], lin_size[5], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[5], affine=do_train_bias),
            nn.SiLU(),
            nn.Dropout(lin_drop[4]),
            nn.Linear(lin_size[5], lin_size[6], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[6], affine=do_train_bias),
            # nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        val = self.classifier(x)
        return val, argmax(val, dim=1)


class cnn_rgc_onoff_v1(__model_settings_common):
    """Classification model"""
    def __init__(self, input_size=32, output_size=4):
        super().__init__('Classifier')
        self.model_embedded = False
        self.model_shape = (1, input_size)
        kernel_layer = [28, 40, 24, 16, 12]
        kernel_size = [4, 4, 3, 3, 2]
        kernel_stride = [1, 1, 1, 1, 1]
        kernel_padding = [3, 3, 2, 2, 1]
        pool_size = [2, 2, 2, 2, 2]
        lin_size = [24, 20, 12, output_size]
        do_train_bias = True

        self.cnn = nn.Sequential(
            nn.Dropout(0.0),
            nn.Conv1d(1, kernel_layer[0], kernel_size[0],
                      stride=kernel_stride[0], padding=kernel_padding[0], bias=do_train_bias),
            nn.BatchNorm1d(kernel_layer[0], affine=do_train_bias),
            nn.PReLU(),
            nn.MaxPool1d(pool_size[0]),
            nn.Conv1d(kernel_layer[0], kernel_layer[1], kernel_size[1],
                      stride=kernel_stride[1], padding=kernel_padding[1], bias=do_train_bias),
            nn.BatchNorm1d(kernel_layer[1], affine=do_train_bias),
            nn.PReLU(),
            nn.MaxPool1d(pool_size[1]),
            nn.Conv1d(kernel_layer[1], kernel_layer[2], kernel_size[2],
                      stride=kernel_stride[2], padding=kernel_padding[2], bias=do_train_bias),
            nn.BatchNorm1d(kernel_layer[2], affine=do_train_bias),
            nn.PReLU(),
            nn.MaxPool1d(pool_size[2]),
            nn.Conv1d(kernel_layer[2], kernel_layer[3], kernel_size[3],
                      stride=kernel_stride[3], padding=kernel_padding[3], bias=do_train_bias),
            nn.BatchNorm1d(kernel_layer[3], affine=do_train_bias),
            nn.PReLU(),
            nn.MaxPool1d(pool_size[3]),
            nn.Conv1d(kernel_layer[3], kernel_layer[4], kernel_size[4],
                      stride=kernel_stride[4], padding=kernel_padding[4], bias=do_train_bias),
            nn.BatchNorm1d(kernel_layer[4], affine=do_train_bias),
            nn.PReLU(),
            nn.MaxPool1d(pool_size[4]),
            nn.Flatten(start_dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(lin_size[0], lin_size[1], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[1], affine=do_train_bias),
            nn.PReLU(),
            nn.Linear(lin_size[1], lin_size[2], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[2], affine=do_train_bias),
            nn.PReLU(),
            nn.Linear(lin_size[2], lin_size[3], bias=do_train_bias),
            nn.BatchNorm1d(lin_size[3], affine=do_train_bias),
            # nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x0 = unsqueeze(x, dim=1)
        val = self.cnn(x0)
        val = self.classifier(val)
        return val, argmax(val, dim=1)


Recommended_Config_PytorchSettings = Config_PyTorch(
    model=dnn_rgc_v1(),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    patience=10,
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
