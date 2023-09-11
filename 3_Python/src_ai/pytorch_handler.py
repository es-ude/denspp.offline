import dataclasses
import os.path
import shutil
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from src.metric import calculate_snr
from elasticai.creator.file_generation.on_disk_path import OnDiskPath


# TODO: Einfügen der Trainings-Auswahl durch model.out_modeltyp
class ConfigPyTorch_Template:
    """Template for configurating pytorch for training a model"""
    def __init__(self):
        # Settings of Models/Training
        # self.model = ai_module.cnn_ae_v1
        self.model = "model name"
        self.is_embedded = False
        self.loss_fn = torch.nn.MSELoss()
        self.num_epochs = 1000
        self.batch_size = 512
        # Settings of Datasets
        self.data_path = 'data'
        self.data_file_name = '2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted'
        self.data_split_ratio = 0.2
        self.data_do_shuffle = True
        self.data_do_augmentation = True
        self.data_num_augmentation = 2000
        self.data_do_normalization = False
        self.data_do_addnoise_cluster = False
        # Dataset Preparation
        self.data_exclude_cluster = [1]
        self.data_sel_pos = []

    def set_optimizer(self, model):
        return torch.optim.Adam(model.parameters())


class training_pytorch:
    """Class for Handling Training of Deep Neural Networks in PyTorch"""
    def __init__(self, type: str, model_name: str, do_train=True) -> None:
        self.__time_start = None
        self.__time_end = None
        self.device = None
        self.os_type = None
        self.__setup_device()

        # --- Saving options
        self.index_folder = 'train' if do_train else 'inference'
        self.embedded_model = False
        self.aitype = type
        self.model_name = model_name
        self.__path2run = 'runs'
        self.__path2log = str()
        self.path2save = str()

        # --- Training input
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.valid_loader = None
        self.num_epoch = 1

    def __setup_device(self) -> None:
        """Setup PyTorch for Training"""
        os_type0 = os.name
        device0 = "CUDA" if torch.cuda.is_available() else "CPU"
        if device0 == "CUDA":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if os_type0 == 'nt':
            self.os_type = 'Windows'
        elif os_type0 == 'posix':
            self.os_type = 'Linux'
        else:
            self.os_type = 'Mac'

        print(f"... using PyTorch with {device0} device on {self.os_type}")

    def __init_train(self) -> None:
        """Do init of class for training"""
        folder_name = '{}_'.format(datetime.now().strftime('%Y%m%d_%H%M%S')) + self.index_folder + '_' + self.model_name

        self.path2save = os.path.join(self.__path2run, folder_name)
        self.__path2log = os.path.join(self.__path2run, folder_name, 'logs')
        self.__writer = SummaryWriter(self.__path2log)

    def load_data(self, training_loader, validation_loader) -> None:
        """Loading data for training and validation in DataLoader format into class"""
        self.train_loader = training_loader
        self.valid_loader = validation_loader

    def load_model(self, model: nn.Module, optimizer, loss_fn, epochs: int, print_model=True) -> None:
        """Loading model, optimizer, loss_fn into class"""
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epoch = epochs
        self.embedded_model = model.out_embedded
        if print_model:
            summary(self.model, input_size=self.model.model_shape)

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0

        self.model.train(True)
        for tdata in self.train_loader:
            self.optimizer.zero_grad()
            data_in = tdata['in']
            data_out = tdata['out']
            _, pred_out = self.model(data_in)
            loss = self.loss_fn(pred_out, data_out)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total_batches += 1

        train_loss = train_loss / total_batches

        return train_loss

    def __do_valid_epoch(self) -> float:
        """Do validation during epoch of training"""
        total_batches = 0
        valid_loss = 0.0

        self.model.eval()
        for vdata in self.valid_loader:
            data_in = vdata['in']
            data_out = vdata['out']
            _, pred_out = self.model(data_in)
            valid_loss += self.loss_fn(pred_out, data_out)
            total_batches += 1

        valid_loss = valid_loss / total_batches

        return valid_loss

    def __do_snr_epoch(self) -> list:
        """Do metric calculation during validation step of training"""
        metric_epoch = []
        self.model.eval()
        for vdata in self.valid_loader:
            data_in = vdata['in']
            data_mean = vdata['mean'].detach().numpy()
            _, pred_out = self.model(data_in)

            snr_in = calculate_snr(data_in.detach().numpy(), data_mean)
            snr_out = calculate_snr(pred_out.detach().numpy(), data_mean)
            metric_epoch.append([snr_out - snr_in])

        metric_epoch = np.array(metric_epoch)
        return [np.min(metric_epoch), np.mean(metric_epoch), np.max(metric_epoch)]

    def do_training(self):
        """Start model training incl. validation and custom-own metric calculation"""
        best_vloss = 1_000_000.

        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S.%f')
        self.__init_train()

        own_metric = []
        model_path = str()

        print(f'\nTraining starts on: {timestamp_string}')
        for epoch in range(0, self.num_epoch):
            loss_train = self.__do_training_epoch()
            loss_valid = self.__do_valid_epoch()

            print(f'... results of epoch {epoch + 1}/{self.num_epoch} [{(epoch + 1) / self.num_epoch * 100:.2f} %]: '
                  f'train_loss = {loss_train:.5f},\tvalid_loss = {loss_valid:.5f}')

            # Log the running loss averaged per batch for both training and validation
            self.__writer.add_scalar('Loss_train', loss_train)
            self.__writer.add_scalar('Loss_valid', loss_valid, epoch+1)
            self.__writer.flush()

            # Track best performance, and save the model
            if loss_valid < best_vloss:
                best_vloss = loss_valid
                model_path = os.path.join(self.__path2log, 'model_{}'.format(epoch))
                torch.save(self.model, model_path)

            # Calculation of custom metrics
            own_metric.append(self.__do_snr_epoch())

        # --- Ausgabe nach Training
        own_metric = np.array(own_metric)
        timestamp_end = datetime.now()
        timestamp_string = timestamp_end.strftime('%H:%M:%S.%f')
        diff_time = timestamp_end - timestamp_start
        diff_string = diff_time

        print(f'Training ends on: {timestamp_string}')
        print(f'Training runs: {diff_string}')
        print(f'\nSave best model: {model_path}')

        shutil.copy(model_path, self.path2save)

        return own_metric

    # TODO: Einfügen der Generierung von VHDL-Files
    def generate_vhdl_file(self):
        """Generating the VHDL code for FPGA implementation"""
        destination = OnDiskPath(os.path.join(self.__path2run, "build"))
        print(f"... generate VHDL output file in folder: {destination}")
        # design = self.model.translate("my_model")
        # design.save_to(destination)
