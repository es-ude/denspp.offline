from torch import nn
from package.dnn.pytorch_control import Config_PyTorch, Config_Dataset
import package.dnn.models.autoencoder_cnn as models_ae
import package.dnn.models.autoencoder_class as models_class
import package.dnn.models.autoencoder_embedded as models_ae_embedded

config_train_ae = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_ae_embedded.dnn_ae_v2(32, 8),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_train_class = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_class.classifier_ae_v1(8, 6),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    #data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_file_name='2023-06-30_Dataset03_SimDaten_Quiroga2020_Sorted',
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=10000,
    data_do_normalization=True,
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)
