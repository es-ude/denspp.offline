import torch
from torch import nn, Tensor, argmax
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset


class mlp_class_v1(nn.Module):
    """Class of an autoencoder with Dense-Layer for feature extraction"""
    def __init__(self):
        super().__init__()
        self.out_modelname = 'mnist_class_v1'
        self.out_modeltyp = 'Classifier'
        self.model_shape = (1, 28, 28)
        self.model_embedded = False
        # --- Settings of model
        do_train_bias = True
        do_train_batch = True
        config_network = [784, 100, 10]

        # --- Model Deployment
        self.model = nn.Sequential()
        for idx, layer_size in enumerate(config_network[1:], start=1):
            self.model.add_module(f"linear_{idx:02d}", nn.Linear(in_features=config_network[idx-1],
                                                                 out_features=layer_size, bias=do_train_bias))
            self.model.add_module(f"batch1d_{idx:02d}", nn.BatchNorm1d(num_features=layer_size, affine=do_train_batch))
            if not idx == len(config_network)-1:
                self.model.add_module(f"act_{idx:02d}", nn.ReLU())
            else:
                self.model.add_module(f"soft", nn.Softmax())

    def forward(self, x: Tensor) -> [Tensor, Tensor]:
        x = torch.flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)


# --- Recommended Configurations for Training
Recommended_Config_Classifier = Config_PyTorch(
    model=mlp_class_v1(),
    loss='Cross Entropy Loss',
    loss_fn=nn.CrossEntropyLoss(),
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
