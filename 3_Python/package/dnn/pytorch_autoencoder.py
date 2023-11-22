import numpy as np
from os.path import join
from shutil import copy
from datetime import datetime
from torch import load, save
from package.dnn.pytorch_control import Config_PyTorch, training_pytorch
from package.metric import calculate_snr


class pytorch_train(training_pytorch):
    """Class for Handling Training of Autoencoders"""
    def __init__(self, config_train: Config_PyTorch, do_train=True) -> None:
        training_pytorch.__init__(self, config_train, do_train)

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0

        self.model.train(True)
        for tdata in self.train_loader[self._run_kfold]:
            self.optimizer.zero_grad()
            pred_out = self.model(tdata['in'])[1]
            loss = self.loss_fn(pred_out, tdata['out'])
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
        for vdata in self.valid_loader[self._run_kfold]:
            pred_out = self.model( vdata['in'])[1]
            valid_loss += self.loss_fn(pred_out, vdata['out']).item()
            total_batches += 1

        valid_loss = valid_loss / total_batches
        return valid_loss

    def __do_snr_epoch(self) -> np.ndarray:
        """Do metric calculation during validation step of training"""
        self.model.eval()
        snr_in = np.zeros(shape=(self._samples_valid[self._run_kfold], ), dtype=float)
        snr_out = np.zeros(shape=(self._samples_valid[self._run_kfold], ), dtype=float)
        run_idx = 0
        for vdata in self.valid_loader[self._run_kfold]:
            data_mean = vdata['mean'].detach().numpy()
            pred_out = self.model(vdata['in'])[1].detach().numpy()

            for idx, data in enumerate(vdata['in'].detach().numpy()):
                snr_in[run_idx] = calculate_snr(data, data_mean[idx, :])
                snr_out[run_idx] = calculate_snr(pred_out[idx, :], data_mean[idx, :])
                run_idx += 1

        return snr_out - snr_in

    def do_training(self) -> list:
        """Start model training incl. validation and custom-own metric calculation"""
        self._init_train()
        self._save_config_txt()
        # --- Handling Kfold cross validation training
        if self._do_kfold:
            print(f"Starting Kfold cross validation training in {self.settings.num_kfold} steps")

        metrics_own = list()
        path2model = str()
        path2model_init = join(self._path2save, f'model_reset.pth')
        save(self.model.state_dict(), path2model_init)
        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S')
        print(f'\nTraining starts on {timestamp_string}')

        for fold in np.arange(self.settings.num_kfold):
            best_loss = np.array((1_000_000., 1_000_000.), dtype=float)
            # Init fold
            epoch_metric = list()
            self.model.load_state_dict(load(path2model_init))
            self._run_kfold = fold
            self._init_writer()

            if self._do_kfold:
                print(f'\nStarting with Fold #{fold}')

            for epoch in range(0, self.settings.num_epochs):
                train_loss = self.__do_training_epoch()
                valid_loss = self.__do_valid_epoch()

                print(f'... results of epoch {epoch + 1}/{self.settings.num_epochs} '
                      f'[{(epoch + 1) / self.settings.num_epochs * 100:.2f} %]: '
                      f'train_loss = {train_loss:.5f},'
                      f'\tvalid_loss = {valid_loss:.5f}')

                # Log the running loss averaged per batch for both training and validation
                self._writer.add_scalar('Loss_train (AE)', train_loss, epoch+1)
                self._writer.add_scalar('Loss_valid (AE)', valid_loss, epoch+1)
                self._writer.flush()

                # Tracking the best performance and saving the model
                if valid_loss < best_loss[1]:
                    best_loss = [train_loss, valid_loss]
                    path2model = join(self._path2temp, f'model_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)

                # Saving metrics after each epoch
                epoch_metric.append(self.__do_snr_epoch())

            # --- Saving metrics after each fold
            metrics_own.append(epoch_metric)
            copy(path2model, self._path2save)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')

        # --- Ending of all trainings phases
        self._end_training_routine(timestamp_start)

        return metrics_own
