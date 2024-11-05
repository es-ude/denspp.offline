import numpy as np
from copy import deepcopy
from os.path import join
from shutil import copy
from datetime import datetime
from torch import Tensor, is_tensor, load, save, inference_mode, flatten, cuda, cat, sub, concatenate
from torch import max, min, log10, sum, randn
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset, training_pytorch


def _calculate_snr(data: Tensor, mean: Tensor) -> Tensor:
    """Calculating the Signal-to-Noise (SNR) ratio of the input data
    Args:
        data:   Tensor with raw data / frame
        mean:   Tensor with class-specific mean data / frame
    Return:
        Tensor with SNR value
    """
    max_values, _ = max(mean, dim=1)
    min_values, _ = min(mean, dim=1)
    a0 = (max_values - min_values) ** 2
    b0 = sum((data - mean) ** 2, dim=1)
    return 10 * log10(a0 / b0)


def _calculate_snr_waveform(input_waveform: Tensor, mean_waveform: Tensor) -> Tensor:
    """Calculation of metric Signal-to-Noise ratio (SNR) of defined input and reference waveform
    Args:
        input_waveform:     Tensor array with input waveform
        mean_waveform:      Tensor array with real mean waveform from dataset
    Return:
        Tensor with differential Signal-to-Noise ratio (SNR) of applied waveforms
    """
    return _calculate_snr(input_waveform, mean_waveform)


def _calculate_dsnr_waveform(input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor) -> Tensor:
    """Calculation of metric different Signal-to-Noise ratio (SNR) between defined input and predicted to reference waveform
    Args:
        input_waveform:     Tensor array with input waveform
        pred_waveform:      Tensor array with predicted waveform from model
        mean_waveform:      Tensor array with real mean waveform from dataset
    Return:
        Tensor with differential Signal-to-Noise ratio (SNR) of applied waveforms
    """
    snr_in = _calculate_snr(input_waveform, mean_waveform)
    snr_out = _calculate_snr(pred_waveform, mean_waveform)
    return sub(snr_out, snr_in)


class train_nn(training_pytorch):
    def __init__(self, config_train: Config_PyTorch, config_data: Config_Dataset, do_train=True, do_print=True) -> None:
        """Class for Handling Training of Autoencoders
        Args:
            config_data:            Settings for handling and loading the dataset (just for saving)
            config_train:           Settings for handling the PyTorch Trainings Routine
            do_train:               Do training of model otherwise only inference
            do_print:               Printing the state and results into Terminal
        Return:
            None
        """
        training_pytorch.__init__(self, config_train, config_data, do_train, do_print)
        # --- Structure for calculating custom metrics during training
        self.__metric_buffer = dict()
        self.__metric_result = dict()
        self._metric_methods = {'snr_in': self.__determine_snr_input, 'snr_in_cl': self.__determine_snr_input_class,
                                'snr_out': self.__determine_snr_output, 'snr_out_cl': self.__determine_snr_output_class,
                                'dsnr_all': self.__determine_dsnr_all, 'dsnr_cl': self.__determine_dsnr_class}

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training
        Return:
            Floating value with training loss value
        """
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

    def __do_valid_epoch(self, epoch_custom_metrics: list) -> float:
        """Do validation during epoch of training
        Args:
            epoch_custom_metrics:   List with entries of custom-made metric calculations
        Return:
            Floating value with validation loss value
        """
        total_batches = 0
        valid_loss = 0.0

        self.model.eval()
        with inference_mode():
            for vdata in self.valid_loader[self._run_kfold]:
                data_x = vdata['in'].to(self.used_hw_dev)
                data_y = vdata['out'].to(self.used_hw_dev)
                data_m = vdata['mean'].to(self.used_hw_dev)
                data_id = vdata['class'].to(self.used_hw_dev)
                data_p = self.model(data_x)[1]

                total_batches += 1
                if len(data_y) > 2:
                    valid_loss += self.loss_fn(flatten(data_p, 1), flatten(data_y, 1)).item()
                else:
                    valid_loss += self.loss_fn(data_p, data_y).item()

                # --- Calculating custom made metrics
                for metric_used in epoch_custom_metrics:
                    self._determine_epoch_metrics(metric_used)(data_x, data_p, data_m, metric_used, data_id)

        return float(valid_loss / total_batches)

    def __process_epoch_metrics_calculation(self, init_phase: bool, custom_made_metrics: list) -> None:
        """Function for preparing the custom-made metric calculation
        Args:
            init_phase:         Boolean decision if processing part is in init (True) or in post-training phase (False)
            custom_made_metrics:List with custom metrics for calculation during validation phase
        Return:
            None
        """
        # --- Init phase for generating empty data structure
        if init_phase:
            for key0 in custom_made_metrics:
                # --- Checking if called metric calculation is in method available
                method_avai = False
                for key1 in self._metric_methods.keys():
                    if key0 == key1:
                        method_avai = True
                        break
                # --- Generating dummy
                if method_avai:
                    self.__metric_result.update({key0: list()})
                    self.__metric_buffer.update({key0: []})
                else:
                    raise NotImplementedError(f"Used custom metric ({key0}) is not implemented - Please check!")
        # --- Processing results
        else:
            for key0 in self.__metric_buffer.keys():
                self.__metric_result[key0].append(self.__metric_buffer[key0])
                self.__metric_buffer.update({key0: []})

    def __determine_snr_input(self, input_waveform: Tensor, pred_waveform: Tensor,
                              mean_waveform: Tensor, *args) -> None:
        """Calculation of SNR in each epoch using validation dataset
        Args:
            input_waveform:     Tensor array with input waveform
            pred_waveform:      Tensor array with predicted waveform from model
            mean_waveform:      Tensor array with real mean waveform from dataset
        Return:
            None
        """
        out = _calculate_snr_waveform(input_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[args[0]], list):
            self.__metric_buffer[args[0]] = out
        else:
            self.__metric_buffer[args[0]] = concatenate((self.__metric_buffer[args[0]], out), dim=0)

    def __determine_snr_input_class(self, input_waveform: Tensor, pred_waveform: Tensor,
                              mean_waveform: Tensor, *args) -> None:
        """Calculation of SNR in each epoch using validation dataset
        Args:
            input_waveform:     Tensor array with input waveform
            pred_waveform:      Tensor array with predicted waveform from model
            mean_waveform:      Tensor array with real mean waveform from dataset
        Return:
            None
        """
        out = self._separate_classes_from_label(
            pred=_calculate_snr_waveform(input_waveform, mean_waveform),
            true=args[1]
        )
        if len(self.__metric_buffer[args[0]]) == 0:
            self.__metric_buffer[args[0]] = out[0]
        else:
            for idx, snr_class in enumerate(out[0]):
                self.__metric_buffer[args[0]][idx].extend(snr_class)

    def __determine_snr_output(self, input_waveform: Tensor, pred_waveform: Tensor,
                              mean_waveform: Tensor, *args) -> None:
        """Calculation of SNR in each epoch using validation dataset
        Args:
            input_waveform:     Tensor array with input waveform
            pred_waveform:      Tensor array with predicted waveform from model
            mean_waveform:      Tensor array with real mean waveform from dataset
        Return:
            None
        """
        out = _calculate_snr_waveform(pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[args[0]], list):
            self.__metric_buffer[args[0]] = out
        else:
            self.__metric_buffer[args[0]] = concatenate((self.__metric_buffer[args[0]], out), dim=0)

    def __determine_snr_output_class(self, input_waveform: Tensor, pred_waveform: Tensor,
                              mean_waveform: Tensor, *args) -> None:
        """Calculation of SNR in each epoch using validation dataset
        Args:
            input_waveform:     Tensor array with input waveform
            pred_waveform:      Tensor array with predicted waveform from model
            mean_waveform:      Tensor array with real mean waveform from dataset
        Return:
            None
        """
        out = self._separate_classes_from_label(
            pred=_calculate_snr_waveform(pred_waveform, mean_waveform),
            true=args[1]
        )
        if len(self.__metric_buffer[args[0]]) == 0:
            self.__metric_buffer[args[0]] = out[0]
        else:
            for idx, snr_class in enumerate(out[0]):
                self.__metric_buffer[args[0]][idx].extend(snr_class)

    def __determine_dsnr_all(self, input_waveform: Tensor, pred_waveform: Tensor,
                             mean_waveform: Tensor, *args) -> None:
        """Calculation of dSNR in each epoch using validation dataset
        Args:
            input_waveform:     Tensor array with input waveform
            pred_waveform:      Tensor array with predicted waveform from model
            mean_waveform:      Tensor array with real mean waveform from dataset
        Return:
            None
        """
        out = _calculate_dsnr_waveform(input_waveform, pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[args[0]], list):
            self.__metric_buffer[args[0]] = out
        else:
            self.__metric_buffer[args[0]] = concatenate((self.__metric_buffer[args[0]], out), dim=0)

    def __determine_dsnr_class(self, input_waveform: Tensor, pred_waveform: Tensor,
                               mean_waveform: Tensor, *args) -> None:
        """Calculation of class-specific dSNR in each epoch using validation dataset
        Args:
            input_waveform:     Tensor array with input waveform
            pred_waveform:      Tensor array with predicted waveform from model
            mean_waveform:      Tensor array with real mean waveform from dataset
        Return:
            None
        """
        out = self._separate_classes_from_label(
            pred=_calculate_dsnr_waveform(input_waveform, pred_waveform, mean_waveform),
            true=args[1]
        )
        if len(self.__metric_buffer[args[0]]) == 0:
            self.__metric_buffer[args[0]] = out[0]
        else:
            for idx, snr_class in enumerate(out[0]):
                self.__metric_buffer[args[0]][idx].extend(snr_class)

    def do_training(self, path2save='', metrics=()) -> dict:
        """Start model training incl. validation and custom-own metric calculation
        Args:
            path2save:      Path for saving the results [Default: '' --> generate new folder]
            metrics:        List with strings of used metric [Default: empty]
        Returns:
            Dictionary with metrics from training (loss_train, loss_valid, own_metrics)
        """
        self._init_train(path2save=path2save, addon='_AE')
        self._save_config_txt('_ae')

        # --- Handling Kfold cross validation training
        if self._kfold_do and self._do_print_state:
            print(f"Starting Kfold cross validation training in {self.settings_train.num_kfold} steps")

        metric_out = dict()
        path2model = str()
        path2model_init = join(self._path2save, f'model_ae_reset.pth')
        save(self.model.state_dict(), path2model_init)
        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S')
        if self._do_print_state:
            print(f'\nTraining starts on {timestamp_string}'
                  f"\n=====================================================================================")

        self.__process_epoch_metrics_calculation(True, metrics)
        for fold in np.arange(self.settings_train.num_kfold):
            # --- Init fold
            best_loss = np.array((1_000_000., 1_000_000.), dtype=float)
            patience_counter = self.settings_train.patience
            metric_fold = dict()
            epoch_loss_train = list()
            epoch_loss_valid = list()
            epoch_metric = [[] for _ in metrics]

            self.model.load_state_dict(load(path2model_init))
            self._run_kfold = fold
            self._init_writer()

            if self._kfold_do and self._do_print_state:
                print(f'\nStarting with Fold #{fold}')

            for epoch in range(0, self.settings_train.num_epochs):
                if self.settings_train.deterministic_do:
                    self.deterministic_generator.manual_seed(self.settings_train.deterministic_seed + epoch)

                loss_train = self.__do_training_epoch()
                loss_valid = self.__do_valid_epoch(metrics)
                epoch_loss_train.append(loss_train)
                epoch_loss_valid.append(loss_valid)
                self.__process_epoch_metrics_calculation(False, metrics)

                if self._do_print_state:
                    print(f'... results of epoch {epoch + 1}/{self.settings_train.num_epochs} '
                          f'[{(epoch + 1) / self.settings_train.num_epochs * 100:.2f} %]: '
                          f'train_loss = {loss_train:.5f},'
                          f'\tvalid_loss = {loss_valid:.5f},'
                          f'\tdelta_loss = {loss_train - loss_valid:.6f}')

                # Log the running loss averaged per batch for both training and validation
                self._writer.add_scalar('Loss_train (AE)', loss_train, epoch + 1)
                self._writer.add_scalar('Loss_valid (AE)', loss_valid, epoch + 1)
                self._writer.flush()

                # Tracking the best performance and saving the model
                if loss_valid < best_loss[1]:
                    best_loss = [loss_train, loss_valid]
                    path2model = join(self._path2temp, f'model_ae_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)
                    patience_counter = self.settings_train.patience
                else:
                    patience_counter -= 1

                # Early Stopping
                if patience_counter <= 0:
                    if self._do_print_state:
                        print(f"... training stopped due to no change after {epoch + 1} epochs!")
                    break

            copy(path2model, self._path2save)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')

            # --- Saving results
            metric_fold.update({'loss_train': epoch_loss_train, 'loss_valid': epoch_loss_valid})
            metric_fold.update(self.__metric_result)
            metric_out.update({f"fold_{fold:03d}": metric_fold})

        # --- Ending of all trainings phases
        self._end_training_routine(timestamp_start)

        # --- Metric out for saving (converting from tensor to numpy)
        metric_save = deepcopy(metric_out)
        for key0, data0 in metric_out.items():
            for key1, data1 in data0.items():
                for idx, data2 in enumerate(data1):
                    if is_tensor(data2):
                        metric_save[key0][key1][idx] = data2.cpu().detach().numpy()

        np.save(f"{self._path2save}/metric_cl", metric_save, allow_pickle=True)
        return metric_out

    def do_validation_after_training(self) -> dict:
        """Performing the validation with the best model after training for plotting and saving results"""
        if cuda.is_available():
            cuda.empty_cache()

        # --- Do the Inference with Best Model
        path2model = self.get_best_model('ae')[0]
        if self._do_print_state:
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
                                               {'feat': result_feat, 'pred': result_pred}, addon='ae')
