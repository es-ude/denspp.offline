from os import remove
from os.path import join, exists
from glob import glob
import shutil
import numpy as np
from datetime import datetime

from torch import load, save
from package.dnn.pytorch_control import Config_PyTorch, training_pytorch


class pytorch_train(training_pytorch):
    """Class for Handling the Training of Classifiers"""
    def __init__(self, config_train: Config_PyTorch, do_train=True) -> None:
        training_pytorch.__init__(self, config_train, do_train)

    def __do_training_epoch(self) -> [float, float]:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0

        self.model.train(True)
        for tdata in self.train_loader[self._run_kfold]:
            self.optimizer.zero_grad()
            pred_cl, dec_cl = self.model(tdata['in'])
            loss = self.loss_fn(pred_cl, tdata['out'])
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total_batches += 1
            total_correct += int(sum(dec_cl == tdata['out']))
            total_samples += len(tdata['in'])

        train_acc = total_correct / total_samples
        train_loss = train_loss / total_batches

        return train_loss, train_acc

    def __do_valid_epoch(self) -> [float, float]:
        """Do validation during epoch of training"""
        valid_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0

        self.model.eval()
        for vdata in self.valid_loader[self._run_kfold]:
            pred_cl, dec_cl = self.model(vdata['in'])

            valid_loss += self.loss_fn(pred_cl, vdata['out']).item()
            total_batches += 1
            total_correct += int(sum(dec_cl == vdata['out']))
            total_samples += len(vdata['in'])

        valid_acc = total_correct / total_samples
        valid_loss = valid_loss / total_batches

        return valid_loss, valid_acc

    def do_training(self) -> [list, list]:
        """Start model training incl. validation and custom-own metric calculation"""
        self._init_train()
        self._save_config_txt()
        # --- Handling Kfold cross validation training
        if self._do_kfold:
            print(f"Starting Kfold cross validation training in {self.settings.num_kfold} steps")

        metrics = list()
        own_metric = list()
        path2model = str()
        path2model_init = join(self._path2save, f'model_reset.pth')
        save(self.model.state_dict(), path2model_init)
        for fold in np.arange(self.settings.num_kfold):
            # Reseting the model
            self.model.load_state_dict(load(path2model_init))
            self._run_kfold = fold
            self._init_writer()

            timestamp_start = datetime.now()
            timestamp_string = timestamp_start.strftime('%H:%M:%S')
            if self._do_kfold:
                print(f'\nTraining starts on: {timestamp_string} with fold #{fold}')
            else:
                print(f'\nTraining starts on: {timestamp_string}')

            best_loss = [1e6, 1e6]
            best_acc = [0.0, 0.0]
            for epoch in range(0, self.settings.num_epochs):
                train_loss, train_acc = self.__do_training_epoch()
                valid_loss, valid_acc = self.__do_valid_epoch()

                print(f'... results of epoch {epoch + 1}/{self.settings.num_epochs} '
                      f'[{(epoch + 1) / self.settings.num_epochs * 100:.2f} %]: '
                      f'train_loss = {train_loss:.5f}, train_acc = {100 * train_acc:.2f} % - '
                      f'valid_loss = {valid_loss:.5f}, valid_acc = {100 * valid_acc:.2f} %')

                # Log the running loss averaged per batch for both training and validation
                self._writer.add_scalar('Loss_train', train_loss)
                self._writer.add_scalar('Loss_valid', valid_loss)
                self._writer.add_scalar('Acc_train', train_acc)
                self._writer.add_scalar('Acc_valid', valid_acc, epoch+1)
                self._writer.flush()

                # Tracking the best performance and saving the model
                if valid_loss < best_loss[1]:
                    best_loss = [train_loss, valid_loss]
                    best_acc = [train_acc, valid_acc]
                    path2model = join(self._path2log, f'model_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)

                # Saving metrics
                own_metric.append(np.array((train_acc, valid_acc), dtype=float))

            # --- Ausgabe nach Training
            metrics.append(best_loss)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')
            self._save_train_results(best_acc[0], best_acc[1], 'Acc.')

            timestamp_end = datetime.now()
            timestamp_string = timestamp_end.strftime('%H:%M:%S')
            diff_time = timestamp_end - timestamp_start
            diff_string = diff_time

            print(f'Training ends on: {timestamp_string}')
            print(f'Training runs: {diff_string}')
            print(f'Save best model: {path2model}')
            shutil.copy(path2model, self._path2save)

        # --- Ending of all trainings phases
        # Delete init model
        if exists(path2model_init):
            remove(path2model_init)

        # Delete log folders
        folder_logs = glob(join(self._path2save, 'logs*'))
        for folder in folder_logs:
            shutil.rmtree(folder, ignore_errors=True)

        return metrics, own_metric
