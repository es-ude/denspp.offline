import numpy as np
from os.path import join
from shutil import copy
from datetime import datetime
from torch import Tensor, load, save, from_numpy, tensor, max, min, log10
from scipy.io import savemat
from package.dnn.pytorch_control import Config_PyTorch, Config_Dataset,  training_pytorch
from package.metric import calculate_snr


class train_nn_autoencoder(training_pytorch):
    """Class for Handling Training of Autoencoders"""
    def __init__(self, config_train: Config_PyTorch, config_dataset: Config_Dataset, do_train=True) -> None:
        training_pytorch.__init__(self, config_train, config_dataset, do_train)

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0

        self.model.train(True)
        for tdata in self.train_loader[self._run_kfold]:
            self.optimizer.zero_grad()
            pred_out = self.model(tdata['in'].to(self.used_hw_dev))[1]
            loss = self.loss_fn(pred_out, tdata['out'].to(self.used_hw_dev))
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
            pred_out = self.model(vdata['in'].to(self.used_hw_dev))[1]
            valid_loss += self.loss_fn(pred_out, vdata['out'].to(self.used_hw_dev)).item()
            pred_out = self.model(vdata['in'].to(self.used_hw_dev))[1]
            valid_loss += self.loss_fn(pred_out, vdata['out'].to(self.used_hw_dev)).item()
            total_batches += 1

        valid_loss = valid_loss / total_batches
        return valid_loss

    def __do_snr_epoch(self) -> Tensor:
        """Do metric calculation during validation step of training"""
        self.model.eval()
        inc_snr = list()
        for vdata in self.valid_loader[self._run_kfold]:
            data_mean = vdata['mean'].to(self.used_hw_dev)
            pred_out = self.model(vdata['in'].to(self.used_hw_dev))[1]
            for idx, data in enumerate(vdata['in'].to(self.used_hw_dev)):
                snr0 = self.__calculate_snr(data, data_mean[idx, :])
                snr1 = self.__calculate_snr(pred_out[idx, :], data_mean[idx, :])
                inc_snr.append(snr1 - snr0)
        return tensor(inc_snr)

    def __calculate_snr(self, yin: Tensor, ymean: Tensor) -> Tensor:
        """Calculating the signal-to-noise ratio [dB] of the input signal compared to mean waveform"""
        a0 = (max(ymean) - min(ymean)) ** 2
        b0 = sum((yin - ymean) ** 2)
        return 10 * log10(a0 / b0)

    def do_training(self, path2save='') -> list:
        """Start model training incl. validation and custom-own metric calculation"""
        self._init_train(path2save=path2save)
        self._save_config_txt('_ae')

        # --- Handling Kfold cross validation training
        if self._do_kfold:
            print(f"Starting Kfold cross validation training in {self.settings.num_kfold} steps")

        run_metric = list()

        path2model = str()
        path2model_init = join(self._path2save, f'model_ae_reset.pth')
        save(self.model.state_dict(), path2model_init)
        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S')
        print(f'\nTraining starts on {timestamp_string}')

        for fold in np.arange(self.settings.num_kfold):
            best_loss = np.array((1_000_000., 1_000_000.), dtype=float)
            # Init fold
            fold_loss = list()
            fold_metric = list()
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
                    path2model = join(self._path2temp, f'model_ae_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)

                # Saving metrics after each epoch
                fold_loss.append((train_loss, valid_loss))
                fold_metric.append(self.__do_snr_epoch())

            # --- Saving metrics after each fold
            run_metric.append((fold_loss, fold_metric))
            copy(path2model, self._path2save)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')

        # --- Ending of all trainings phases
        self._end_training_routine(timestamp_start)

        return run_metric

    def do_validation_after_training(self, num_output=4) -> dict:
        """Performing the validation with the best model after training for plotting and saving results"""

        # --- Getting data from validation set for inference
        data_valid = self.get_data_points(num_output, use_train_dataloader=False)
        data_train = self.get_data_points(num_output, use_train_dataloader=True)

        # --- Do the Inference with Best Model
        print(f"\nDoing the inference with validation data on best model")
        model_test = load(self.get_best_model('ae')[0])
        feat_out, pred_out = model_test(from_numpy(data_valid['in']))
        feat_out = feat_out.detach().numpy()
        pred_out = pred_out.detach().numpy()

        # --- Calculating the improved SNR
        snr_in = []
        snr_out = []
        for idx, _ in enumerate(pred_out):
            snr_in.append(calculate_snr(data_valid['in'][idx, :], data_valid['mean'][idx, :]))
            snr_out.append(calculate_snr(pred_out[idx, :], data_valid['mean'][idx, :]))

        snr_in = np.array(snr_in)
        snr_out = np.array(snr_out)
        snr_inc = snr_out - snr_in

        print(f"\nCalcuted SNR values from inference on validated datas")
        print(f"- SNR_in: {np.median(snr_in):.2f} (median) | {np.max(snr_in):.2f} (max) | {np.min(snr_in):.2f} (min)")
        print(
            f"- SNR_out: {np.median(snr_out):.2f} (median) | {np.max(snr_out):.2f} (max) | {np.min(snr_out):.2f} (min)")
        print(f"- SNR_inc: {np.median(snr_inc): .2f} (median)")

        # --- Producing the output
        output = dict()
        output.update({'settings': self.settings, 'date': datetime.now().strftime('%d/%m/%Y, %H:%M:%S')})
        output.update({'train_clus': data_train['cluster'], 'valid_clus': data_valid['cluster']})
        output.update({'input': data_valid['in'], 'feat': feat_out, 'pred': pred_out})
        output.update({'cl_dict': self.cell_classes})

        # --- Saving dict
        savemat(join(self.get_saving_path(), 'results_ae.mat'), output,
                do_compression=True, long_field_names=True)
        return output
