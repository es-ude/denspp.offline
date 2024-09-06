import numpy as np
from os.path import join
from shutil import copy
from datetime import datetime
from torch import Tensor, from_numpy, load, save, tensor, inference_mode, flatten, cuda, cat
from torch import max, min, log10, sum, randn
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset, training_pytorch


def _calculate_snr(vdata_in: Tensor, vdata_mean: Tensor) -> Tensor:
    """a0 = ((max(ymean) - min(ymean))**2"""
    max_values, _ = max(vdata_mean, dim=1)
    min_values, _ = min(vdata_mean, dim=1)
    a0 = (max_values - min_values) ** 2
    b0 = sum((vdata_in - vdata_mean) ** 2, dim=1)
    result = 10 * log10(a0 / b0)
    return result


class train_nn(training_pytorch):
    """Class for Handling Training of Autoencoders"""
    def __init__(self, config_train: Config_PyTorch, config_data: Config_Dataset, do_train=True) -> None:
        training_pytorch.__init__(self, config_train, config_data, do_train)

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0

        self.model.train(True)
        for tdata in self.train_loader[self._run_kfold]:
            data_x = tdata['in'].to(self.used_hw_dev)
            data_y = tdata['out'].to(self.used_hw_dev)
            data_p = self.model(data_x)[1]

            self.optimizer.zero_grad()
            if len(data_y) > 2:
                loss = self.loss_fn(flatten(data_p, 1), flatten(data_y, 1))
            else:
                loss = self.loss_fn(data_p, data_y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total_batches += 1
        return float(train_loss / total_batches)

    def __do_valid_epoch(self) -> float:
        """Do validation during epoch of training"""
        total_batches = 0
        valid_loss = 0.0

        self.model.eval()
        with inference_mode():
            for vdata in self.valid_loader[self._run_kfold]:
                data_x = vdata['in'].to(self.used_hw_dev)
                data_y = vdata['out'].to(self.used_hw_dev)
                data_p = self.model(data_x)[1]

                total_batches += 1
                if len(data_y) > 2:
                    valid_loss += self.loss_fn(flatten(data_p, 1), flatten(data_y, 1)).item()
                else:
                    valid_loss += self.loss_fn(data_p, data_y).item()
        return float(valid_loss / total_batches)

    def __do_snr_epoch(self) -> Tensor:
        """Do metric calculation during validation step of training"""
        self.model.eval()
        inc_snr = list()
        with inference_mode():
            for vdata in self.valid_loader[self._run_kfold]:
                data_mean = vdata['mean'].to(self.used_hw_dev)
                data_in = vdata['in'].to(self.used_hw_dev)
                pred_out = self.model(vdata['in'].to(self.used_hw_dev))[1]

                snr0_0 = _calculate_snr(data_in, data_mean)
                snr1_0 = _calculate_snr(pred_out, data_mean)
                inc_snr.extend((snr1_0 - snr0_0))
        return tensor(inc_snr)

    def __do_calc_metric(self, used_metrics: str) -> Tensor:
        """Determination of additional metrics during training"""
        out = Tensor()
        match used_metrics:
            case 'snr':
                out = self.__do_snr_epoch()
            case '':
                out = Tensor()

        return out

    def do_training(self, path2save='', metrics='') -> list:
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
        print(f'\nTraining starts on {timestamp_string}'
              f"\n=====================================================================================")

        for fold in np.arange(self.settings.num_kfold):
            # --- Init fold
            best_loss = np.array((1_000_000., 1_000_000.), dtype=float)
            patience_counter = self.settings.patience
            epoch_loss = list()
            epoch_metric = list()
            self.model.load_state_dict(load(path2model_init))
            self._run_kfold = fold
            self._init_writer()

            if self._do_kfold:
                print(f'\nStarting with Fold #{fold}')

            for epoch in range(0, self.settings.num_epochs):
                loss_train = self.__do_training_epoch()
                loss_valid = self.__do_valid_epoch()

                print(f'... results of epoch {epoch + 1}/{self.settings.num_epochs} '
                      f'[{(epoch + 1) / self.settings.num_epochs * 100:.2f} %]: '
                      f'train_loss = {loss_train:.5f},'
                      f'\tvalid_loss = {loss_valid:.5f},'
                      f'\tdelta_loss = {loss_train-loss_valid:.6f}')

                # Log the running loss averaged per batch for both training and validation
                self._writer.add_scalar('Loss_train (AE)', loss_train, epoch+1)
                self._writer.add_scalar('Loss_valid (AE)', loss_valid, epoch+1)
                self._writer.flush()

                # Saving metrics after each epoch
                epoch_loss.append((loss_train, loss_valid))
                epoch_metric.append(self.__do_calc_metric(metrics))

                # Tracking the best performance and saving the model
                if loss_valid < best_loss[1]:
                    best_loss = [loss_train, loss_valid]
                    path2model = join(self._path2temp, f'model_ae_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)
                    patience_counter = self.settings.patience
                else:
                    patience_counter -= 1

                # Early Stopping
                if patience_counter <= 0:
                    print(f"... training stopped due to no change after {epoch+1} epochs!")
                    break

            # --- Saving metrics after each fold
            run_metric.append((epoch_loss, epoch_metric))
            copy(path2model, self._path2save)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')

        # --- Ending of all trainings phases
        self._end_training_routine(timestamp_start)
        return run_metric

    def do_validation_after_training(self) -> dict:
        """Performing the validation with the best model after training for plotting and saving results"""
        if cuda.is_available():
            cuda.empty_cache()

        # --- Do the Inference with Best Model
        path2model = self.get_best_model('ae')[0]
        print("\n================================================================="
              f"\nDo Validation with best model: {path2model}")
        model_test = load(path2model)

        pred_model = randn(32, 1)
        feat_model = randn(32, 1)
        clus_orig_list = randn(32, 1)
        data_orig_list = randn(32, 1)

        first_cycle = True
        for ite_cycle, vdata in enumerate(self.valid_loader[-1]):
            feat, pred = model_test(vdata['in'].to(self.used_hw_dev))
            if first_cycle:
                feat_model = feat.detach().cpu()
                pred_model = pred.detach().cpu()
                clus_orig_list = vdata['class']
                data_orig_list = vdata['in']
            else:
                feat_model = cat((feat_model, feat.detach().cpu()), dim=0)
                pred_model = cat((pred_model, pred.detach().cpu()), dim=0)
                clus_orig_list = cat((clus_orig_list, vdata['class']), dim=0)
                data_orig_list = cat((data_orig_list, vdata['in']), dim=0)
            first_cycle = False

        # --- Preparing output
        result_feat = feat_model.numpy()
        result_pred = pred_model.numpy()
        return self._getting_data_for_plotting(data_orig_list.numpy(), clus_orig_list.numpy(),
                                               {'feat': result_feat, 'pred': result_pred})
