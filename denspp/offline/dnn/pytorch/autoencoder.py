import numpy as np
from os.path import join
from shutil import copy
from datetime import datetime
from torch import Tensor, load, save, inference_mode, flatten, cuda, cat, concatenate, randn, sum, abs

from denspp.offline.dnn.ptq_help import quantize_model_fxp
from denspp.offline.dnn.pytorch_handler import ConfigPytorch, ConfigDataset, PyTorchHandler
from denspp.offline.metric.snr import calculate_snr_tensor_waveform, calculate_dsnr_tensor_waveform


class TrainAutoencoder(PyTorchHandler):
    def __init__(self, config_train: ConfigPytorch, config_data: ConfigDataset,
                 do_train: bool=True, do_print: bool=True) -> None:
        """Class for Handling Training of Autoencoders
        Args:
            config_data:            Settings for handling and loading the dataset (just for saving)
            config_train:           Settings for handling the PyTorch Trainings Routine
            do_train:               Do training of model otherwise only inference
            do_print:               Printing the state and results into Terminal
        Return:
            None
        """
        PyTorchHandler.__init__(self, config_train, config_data, do_train, do_print)
        # --- Structure for calculating custom metrics during training
        self._ptq_level = [12, 8]
        self.__metric_buffer = dict()
        self.__metric_result = dict()
        self._metric_methods = {'snr_in': self.__determine_snr_input, 'snr_in_cl': self.__determine_snr_input_class,
                                'snr_out': self.__determine_snr_output, 'snr_out_cl': self.__determine_snr_output_class,
                                'dsnr_all': self.__determine_dsnr_all, 'dsnr_cl': self.__determine_dsnr_class,
                                'ptq_loss': self.__determine_ptq_loss}

    def define_ptq_level(self, total_bitwidth: int, frac_bitwidth: int) -> None:
        """Function for defining the post-training quantization level of the model
        :param total_bitwidth: Total bitwidth of the model
        :param frac_bitwidth: Fraction of bitwidth used for quantization
        :return: None
        """
        self._ptq_level = [total_bitwidth, frac_bitwidth]

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
        self.total_batches_valid = 0
        valid_loss = 0.0

        self.model.eval()
        with inference_mode():
            for vdata in self.valid_loader[self._run_kfold]:
                data_x = vdata['in'].to(self.used_hw_dev)
                data_y = vdata['out'].to(self.used_hw_dev)
                data_m = vdata['mean'].to(self.used_hw_dev)
                data_id = vdata['class'].to(self.used_hw_dev)
                data_p = self.model(data_x)[1]

                self.total_batches_valid += 1
                if len(data_y) > 2:
                    valid_loss += self.loss_fn(flatten(data_p, 1), flatten(data_y, 1)).item()
                else:
                    valid_loss += self.loss_fn(data_p, data_y).item()

                # --- Calculating custom made metrics
                for metric_used in epoch_custom_metrics:
                    self._determine_epoch_metrics(metric_used)(data_x, data_p, data_m, metric_used, data_id)

        return float(valid_loss / self.total_batches_valid)

    def __process_epoch_metrics_calculation(self, init_phase: bool, custom_made_metrics: list) -> None:
        """Function for preparing the custom-made metric calculation
        Args:
            init_phase:         Boolean decision if processing part is in init (True) or in training phase (False)
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
                    self.__metric_buffer.update({key0: list()})
                else:
                    raise NotImplementedError(f"Used custom metric ({key0}) is not implemented - Please check!")
        # --- Processing results
        else:
            for key0 in self.__metric_buffer.keys():
                self.__metric_result[key0].append(self.__metric_buffer[key0])
                self.__metric_buffer.update({key0: list()})

    def __determine_snr_input(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, *args) -> None:
        out = calculate_snr_tensor_waveform(input_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[args[0]], list):
            self.__metric_buffer[args[0]] = out
        else:
            self.__metric_buffer[args[0]] = concatenate((self.__metric_buffer[args[0]], out), dim=0)

    def __determine_snr_input_class(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, *args) -> None:
        out = self._separate_classes_from_label(
            pred=calculate_snr_tensor_waveform(input_waveform, mean_waveform),
            true=args[1], label=args[0]
        )
        if len(self.__metric_buffer[args[0]]) == 0:
            self.__metric_buffer[args[0]] = out[0]
        else:
            for idx, snr_class in enumerate(out[0]):
                old = self.__metric_buffer[args[0]][idx]
                self.__metric_buffer[args[0]][idx] = concatenate((old, snr_class), dim=0)

    def __determine_snr_output(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, *args) -> None:
        out = calculate_snr_tensor_waveform(pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[args[0]], list):
            self.__metric_buffer[args[0]] = out
        else:
            self.__metric_buffer[args[0]] = concatenate((self.__metric_buffer[args[0]], out), dim=0)

    def __determine_snr_output_class(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, *args) -> None:
        out = self._separate_classes_from_label(
            pred=calculate_snr_tensor_waveform(pred_waveform, mean_waveform),
            true=args[1], label=args[0]
        )
        if len(self.__metric_buffer[args[0]]) == 0:
            self.__metric_buffer[args[0]] = out[0]
        else:
            for idx, snr_class in enumerate(out[0]):
                old = self.__metric_buffer[args[0]][idx][0]
                self.__metric_buffer[args[0]][idx] = concatenate((old, snr_class), dim=0)

    def __determine_dsnr_all(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, *args) -> None:
        out = calculate_dsnr_tensor_waveform(input_waveform, pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[args[0]], list):
            self.__metric_buffer[args[0]] = out
        else:
            self.__metric_buffer[args[0]] = concatenate((self.__metric_buffer[args[0]], out), dim=0)

    def __determine_dsnr_class(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, *args) -> None:
        out = calculate_dsnr_tensor_waveform(input_waveform, pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[args[0]], list):
            self.__metric_buffer[args[0]] = out
        else:
            self.__metric_buffer[args[0]] = concatenate((self.__metric_buffer[args[0]], out), dim=0)

    def __determine_ptq_loss(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, *args) -> None:
        """if not hasattr(self.model, 'bit_config'):
            raise NotImplementedError('PTQ Test is only available with elasticAI.creator Models or '
                                      'model includes variable \"bit_config\" = [total_bitwidth, frac_bitwidth]')
        else:"""
        # --- Load model and make inference
        model_ptq = quantize_model_fxp(self.model, self._ptq_level[0], self._ptq_level[1])
        pred_waveform_ptq = model_ptq(input_waveform)[1]
        model_ptq.eval()

        # --- Calculate loss
        if len(input_waveform) > 2:
            loss = self.loss_fn(flatten(pred_waveform_ptq, 1), flatten(input_waveform, 1)).item() / self.total_batches_valid
        else:
            loss = self.loss_fn(pred_waveform_ptq, input_waveform).item() / self.total_batches_valid

        # --- Saving results
        if len(self.__metric_buffer[args[0]]):
            self.__metric_buffer[args[0]][0] = self.__metric_buffer[args[0]][0] + loss
        else:
            self.__metric_buffer[args[0]].append(loss)


    def do_training(self, path2save='', metrics=()) -> dict:
        """Start model training incl. validation and custom-own metric calculation
        Args:
            path2save:      Path for saving the results [Default: '' --> generate new folder]
            metrics:        List with strings of used metric [Default: empty]
        Returns:
            Dictionary with metrics from training (loss_train, loss_valid, own_metrics)
        """
        self._init_train(path2save=path2save, addon='_AE')

        # --- Handling Kfold cross validation training
        if self._kfold_do and self._do_print_state:
            print(f"Starting Kfold cross validation training in {self.settings_train.num_kfold} steps")

        metric_out = dict()
        path2model = str()
        path2model_init = join(self._path2save, f'model_ae_reset.pt')
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

            self.model.load_state_dict(load(path2model_init, weights_only=False))
            self._run_kfold = fold
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

                # Tracking the best performance and saving the model
                if loss_valid < best_loss[1]:
                    best_loss = [loss_train, loss_valid]
                    path2model = join(self._path2temp, f'model_ae_fold{fold:03d}_epoch{epoch:04d}.pt')
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
        metric_save = self._converting_tensor_to_numpy(metric_out)
        np.save(f"{self._path2save}/metric_ae", metric_save, allow_pickle=True)
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
        model_test = load(path2model, weights_only=False)

        pred_model = randn(32, 1)
        feat_model = randn(32, 1)
        clus_orig_list = randn(32, 1)
        data_orig_list = randn(32, 1)

        first_cycle = True
        model_test.eval()
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

    def do_validation_after_training_ptq(self) -> dict:
        """Performing the validation with the best model after training for plotting and saving results"""
        if cuda.is_available():
            cuda.empty_cache()

        # --- Do the Inference with Best Model
        path2model = self.get_best_model('ae')[0]
        if self._do_print_state:
            print("\n================================================================="
                  f"\nDo Validation with best model: {path2model}")
        model_test = load(path2model, weights_only=False)
        model_quant = quantize_model_fxp(model_test, self._ptq_level[0], self._ptq_level[1])

        pred_model = randn(32, 1)
        feat_model = randn(32, 1)
        clus_orig_list = randn(32, 1)
        data_orig_list = randn(32, 1)

        first_cycle = True
        model_quant.eval()
        for ite_cycle, vdata in enumerate(self.valid_loader[-1]):
            feat, pred = model_quant(vdata['in'].to(self.used_hw_dev))
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
                                               {'feat': result_feat, 'pred': result_pred}, addon='ae_quant')
