import dataclasses
import os.path
from glob import glob
import shutil
import numpy as np
from datetime import datetime
from torch import load, save
from package.dnn.pytorch_control import Config_PyTorch, training_pytorch
from package.metric import calculate_snr


class pytorch_autoencoder(training_pytorch):
    """Class for Handling Training of Autoencoders"""
    def __init__(self, type: str, model_name: str, config_train: Config_PyTorch, do_train=True) -> None:
        training_pytorch.__init__(self, type, model_name, config_train, do_train)

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0

        self.model.train(True)
        for tdata in self.train_loader[self.run_kfold]:
            self.optimizer.zero_grad()
            data_in = tdata['in']
            data_out = tdata['out']
            pred_out = self.model(data_in)[1]
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
        for vdata in self.valid_loader[self.run_kfold]:
            data_in = vdata['in']
            data_out = vdata['out']
            pred_out = self.model(data_in)[1]
            valid_loss += self.loss_fn(pred_out, data_out)
            total_batches += 1

        valid_loss = valid_loss / total_batches

        return valid_loss

    def __do_snr_epoch(self) -> list:
        """Do metric calculation during validation step of training"""
        metric_epoch = []
        self.model.eval()
        for vdata in self.valid_loader[self.run_kfold]:
            data_in = vdata['in']
            data_mean = vdata['mean'].detach().numpy()
            pred_out = self.model(data_in)[1]

            snr_in = calculate_snr(data_in.detach().numpy(), data_mean)
            snr_out = calculate_snr(pred_out.detach().numpy(), data_mean)
            metric_epoch.append([snr_out - snr_in])

        metric_epoch = np.array(metric_epoch)
        return [np.min(metric_epoch), np.mean(metric_epoch), np.max(metric_epoch)]

    def do_training(self) -> [list, list]:
        """Start model training incl. validation and custom-own metric calculation"""
        self.init_train()
        self.save_config_txt()
        # --- Handling Kfold cross validation training
        if self.do_kfold:
            print(f"Starting Kfold cross validation training in {self.settings.num_kfold} steps")

        metrics = list()
        own_metric = list()
        path2model = str()
        path2model_init = os.path.join(self.path2save, f'model_reset.pth')
        save(self.model.state_dict(), path2model_init)
        for fold in np.arange(self.settings.num_kfold):
            best_vloss = 1_000_000.
            loss_train = 1_000_000.
            loss_valid = 1_000_000.
            own_metric = []

            # - Reset of model
            self.model.load_state_dict(load(path2model_init))
            self.run_kfold = fold
            self.init_writer()

            timestamp_start = datetime.now()
            timestamp_string = timestamp_start.strftime('%H:%M:%S')
            if self.do_kfold:
                print(f'\nTraining starts on: {timestamp_string} with fold #{fold}')
            else:
                print(f'\nTraining starts on: {timestamp_string}')

            for epoch in range(0, self.settings.num_epochs):
                loss_train = self.__do_training_epoch()
                loss_valid = self.__do_valid_epoch()

                print(f'... results of epoch {epoch + 1}/{self.settings.num_epochs} [{(epoch + 1) / self.settings.num_epochs * 100:.2f} %]: '
                      f'train_loss = {loss_train:.5f},'
                      f'\tvalid_loss = {loss_valid:.5f}')

                # Log the running loss averaged per batch for both training and validation
                self.writer.add_scalar('Loss_train', loss_train)
                self.writer.add_scalar('Loss_valid', loss_valid, epoch+1)
                self.writer.flush()

                # Tracking the best performance and saving the model
                if loss_valid < best_vloss:
                    best_vloss = loss_valid
                    path2model = os.path.join(self.path2log, f'model_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)

                # Calculation of custom metrics
                # own_metric.append(self.__do_snr_epoch())

            # --- Ausgabe nach Training
            self.save_train_results(loss_train, loss_valid)
            own_metric = np.array(own_metric)
            metrics.append([loss_train, loss_valid])

            timestamp_end = datetime.now()
            timestamp_string = timestamp_end.strftime('%H:%M:%S')
            diff_time = timestamp_end - timestamp_start
            diff_string = diff_time

            print(f'Training ends on: {timestamp_string}')
            print(f'Training runs: {diff_string}')
            print(f'Save best model: {path2model}')
            shutil.copy(path2model, self.path2save)

        # --- Ending of all trainings phases
        # Delete init model
        if os.path.exists(path2model_init):
            os.remove(path2model_init)

        # Delete log folders
        folder_logs = glob(os.path.join(self.path2save, 'logs*'))
        for folder in folder_logs:
            shutil.rmtree(folder, ignore_errors=True)

        return metrics, own_metric


class pytorch_classifier(training_pytorch):
    """Class for Handling the Training of Classifiers"""
    def __init__(self, type: str, model_name: str, config_train: Config_PyTorch, do_train=True) -> None:
        training_pytorch.__init__(self, type, model_name, config_train, do_train)

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0

        self.model.train(True)
        for tdata in self.train_loader[self.run_kfold]:
            self.optimizer.zero_grad()
            data_in = tdata['in']
            data_out = tdata['out']
            pred_out = self.model(data_in)
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
        for vdata in self.valid_loader[self.run_kfold]:
            data_in = vdata['in']
            data_out = vdata['out']
            pred_out = self.model(data_in)
            valid_loss += self.loss_fn(pred_out, data_out)
            total_batches += 1

        valid_loss = valid_loss / total_batches

        return valid_loss

    def __do_snr_epoch(self) -> list:
        """Do metric calculation during validation step of training"""
        metric_epoch = []
        self.model.eval()
        for vdata in self.valid_loader[self.run_kfold]:
            data_in = vdata['in']
            data_mean = vdata['mean'].detach().numpy()
            pred_out = self.model(data_in)

            snr_in = calculate_snr(data_in.detach().numpy(), data_mean)
            snr_out = calculate_snr(pred_out.detach().numpy(), data_mean)
            metric_epoch.append([snr_out - snr_in])

        metric_epoch = np.array(metric_epoch)
        return [np.min(metric_epoch), np.mean(metric_epoch), np.max(metric_epoch)]

    def do_training(self) -> list:
        """Start model training incl. validation and custom-own metric calculation"""
        self.__init_train()
        self.__save_config_txt()
        # --- Handling Kfold cross validation training
        if self.do_kfold:
            print(f"Starting Kfold cross validation training in {self.settings.num_kfold} steps")

        metrics = list()
        own_metric = list()
        path2model = str()
        path2model_init = os.path.join(self.path2save, f'model_reset.pth')
        save(self.model.state_dict(), path2model_init)
        for fold in np.arange(self.settings.num_kfold):
            best_vloss = 1_000_000.
            loss_train = 1_000_000.
            loss_valid = 1_000_000.
            own_metric = []

            # - Reset of model
            self.model.load_state_dict(load(path2model_init))
            self.run_kfold = fold
            self.__init_writer()

            timestamp_start = datetime.now()
            timestamp_string = timestamp_start.strftime('%H:%M:%S')
            if self.do_kfold:
                print(f'\nTraining starts on: {timestamp_string} with fold #{fold}')
            else:
                print(f'\nTraining starts on: {timestamp_string}')

            for epoch in range(0, self.settings.num_epochs):
                loss_train = self.__do_training_epoch()
                loss_valid = self.__do_valid_epoch()

                print(f'... results of epoch {epoch + 1}/{self.settings.num_epochs} [{(epoch + 1) / self.settings.num_epochs * 100:.2f} %]: '
                      f'train_loss = {loss_train:.5f},'
                      f'\tvalid_loss = {loss_valid:.5f}')

                # Log the running loss averaged per batch for both training and validation
                self.__writer.add_scalar('Loss_train', loss_train)
                self.__writer.add_scalar('Loss_valid', loss_valid, epoch+1)
                self.__writer.flush()

                # Tracking the best performance and saving the model
                if loss_valid < best_vloss:
                    best_vloss = loss_valid
                    path2model = os.path.join(self.__path2log, f'model_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)

            # --- Ausgabe nach Training
            self.__save_train_results(loss_train, loss_valid)
            own_metric = np.array(own_metric)
            metrics.append([loss_train, loss_valid])

            timestamp_end = datetime.now()
            timestamp_string = timestamp_end.strftime('%H:%M:%S')
            diff_time = timestamp_end - timestamp_start
            diff_string = diff_time

            print(f'Training ends on: {timestamp_string}')
            print(f'Training runs: {diff_string}')
            print(f'Save best model: {path2model}')
            shutil.copy(path2model, self.path2save)

        # --- Ending of all trainings phases
        # Delete init model
        if os.path.exists(path2model_init):
            os.remove(path2model_init)

        # Delete log folders
        folder_logs = glob(os.path.join(self.path2save, 'logs*'))
        for folder in folder_logs:
            shutil.rmtree(folder, ignore_errors=True)

        return metrics

