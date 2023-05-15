import os.path, shutil
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src_ai.dae_dataset import calculate_snr

class training_pytorch:
    def __init__(self, type: str, model_name: str) -> None:
        self.__time_start = None
        self.__time_end = None
        self.device = None
        self.__setup_device()

        # --- Saving options
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
        os_type = os.name
        device0 = "CUDA" if torch.cuda.is_available() else "CPU"
        if device0 == "CUDA":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"... using PyTorch with {device0} device on {os_type}")

    def __init_train(self) -> None:
        folder_name = '{}_ai_training_'.format(datetime.now().strftime('%Y%m%d_%H%M%S')) + self.model_name

        self.path2save = os.path.join(self.__path2run, folder_name)
        self.__path2log = os.path.join(self.__path2run, folder_name, 'logs')
        self.__writer = SummaryWriter(self.__path2log)

    def load_data(self, training_loader, validation_loader) -> None:
        self.train_loader = training_loader
        self.valid_loader = validation_loader

    def load_model(self, model: nn.Module, optimizer, loss_fn, epochs: int) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epoch = epochs

    def do_training(self):
        epoch_number = 0
        best_vloss = 1_000_000.

        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S.%f')
        self.__init_train()

        own_metric = np.zeros(shape=(self.num_epoch, 3))

        print(f'\nTraining starts on: {timestamp_string}')
        for epoch in range(0, self.num_epoch):
            # --- Training
            train_loss = 0.0
            total_batches = 0
            
            self.model.train(True)
            for tdata in self.train_loader:
                self.optimizer.zero_grad()
                (data_in, data_out, pred_out) = self.__calc_model(tdata)
                loss = self.loss_fn(pred_out, data_out)
                loss.backward()
                self.optimizer.step()

                # Gather data and report
                train_loss += loss.item()
                total_batches += 1

            train_loss = train_loss / total_batches

            # --- Validation
            total_batches = 0
            valid_loss = 0.0
            own_batch = []

            self.model.eval()
            for vdata in self.valid_loader:
                (data_in, data_out, pred_out) = self.__calc_model(vdata)
                valid_loss += self.loss_fn(pred_out, data_out)
                total_batches += 1

                # Code for own metric calculation
                own_batch.append([calculate_snr(pred_out.detach().numpy(), data_out.detach().numpy())])

            own_batch = np.array(own_batch)
            own_metric[epoch, 0] = np.min(own_batch)
            own_metric[epoch, 1] = np.mean(own_batch)
            own_metric[epoch, 2] = np.max(own_batch)

            valid_loss = valid_loss / total_batches
            print(f'... loss of epoch {epoch + 1}/{self.num_epoch} [{(epoch + 1) / self.num_epoch * 100:.2f} %]: train = {train_loss:.5f}, valid = {valid_loss:.5f}')

            # Log the running loss averaged per batch for both training and validation
            self.__writer.add_scalar('Loss_train', train_loss)
            self.__writer.add_scalar('Loss_valid', valid_loss, epoch_number+1)
            self.__writer.flush()

            # Track best performance, and save the model
            if valid_loss < best_vloss:
                best_vloss = valid_loss
                model_path = os.path.join(self.__path2log, 'model_{}'.format(epoch_number))
                torch.save(self.model, model_path)

            epoch_number += 1

        # --- Ausgabe nach Training
        timestamp_end = datetime.now()
        timestamp_string = timestamp_end.strftime('%H:%M:%S.%f')
        diff_time = timestamp_end - timestamp_start
        diff_string = diff_time

        print(f'Training ends on: {timestamp_string}')
        print(f'Training runs: {diff_string}')
        print(f'\nSave best model: {model_path}')

        shutil.copy(model_path, self.path2save)

        return own_metric

    def __calc_model(self, data):
        if self.aitype == "dae":
            din = data['frame']
            dout = data['mean_frame']
            _, pred_out = self.model(din)
        else:
            din = data['frame']
            dout = data['spk_type']
            pred_out = self.model(din)

        return din, dout, pred_out
