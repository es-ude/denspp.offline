from torch import nn, Tensor, argmax
from package.dnn.pytorch_control import Config_PyTorch


class dnn_bae_v1(nn.Module):
    """Class to identify spike type from NEO operator"""
    def __init__(self, input_size=32, output_size=4):
        super().__init__()
        self.out_modelname = 'sda_bae_v1'
        self.out_modeltyp = 'BAE'
        self.model_embedded = False
        self.model_shape = (1, input_size)
        self.out_class = ['Spike', 'Background', 'Artefact', 'Non-Spike']
        do_train_bias = True
        iohiddenlayer = [input_size, 20, 12, output_size]
        use_bias = False

        # --- Encoder Path
        self.classifier = nn.Sequential(
            nn.Linear(in_features=iohiddenlayer[0], out_features=iohiddenlayer[1], bias=use_bias),
            nn.BatchNorm1d(iohiddenlayer[1], affine=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[1], out_features=iohiddenlayer[2], bias=use_bias),
            nn.BatchNorm1d(iohiddenlayer[2], affine=do_train_bias),
            nn.Tanh(),
            nn.Linear(in_features=iohiddenlayer[2], out_features=iohiddenlayer[3], bias=use_bias),
            nn.BatchNorm1d(iohiddenlayer[3], affine=do_train_bias),
            nn.Softmax(dim=0)
        )

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        xdist = self.classifier(x)
        return xdist, argmax(xdist)


Recommended_Config_PytorchSettings = Config_PyTorch(
    model=dnn_bae_v1(),
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