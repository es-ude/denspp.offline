import numpy as np
from logging import getLogger, Logger
from os.path import join
from shutil import copy
from datetime import datetime
from torch import Tensor, load, save, inference_mode, flatten, cuda, cat, concatenate, randn

from denspp.offline import check_keylist_elements_any
from denspp.offline.dnn import SettingsPytorch, DatasetFromFile
from denspp.offline.dnn.ptq_help import quantize_model_fxp
from denspp.offline.dnn.training.common_train import SettingsDataset, PyTorchHandler
from denspp.offline.dnn.training.autoencoder_dataset import DatasetAutoencoder
from denspp.offline.metric.snr import calculate_snr_tensor, calculate_dsnr_tensor


class TrainAutoencoder(PyTorchHandler):
    _logger: Logger

    def __init__(self, config_train: SettingsPytorch, config_data: SettingsDataset, do_train: bool=True) -> None:
        """Class for Handling Training of Autoencoders
        :param config_data:     Settings for handling and loading the dataset (just for saving)
        :param config_train:    Settings for handling the PyTorch Trainings Routine
        :param do_train:        Do training of model otherwise only inference
        :return:                None
        """
        PyTorchHandler.__init__(self, config_train, config_data, do_train)
        self._logger = getLogger(__name__)
        self.__metric_buffer = dict()
        self.__metric_result = dict()
        self._metric_methods = {'snr_in': self.__determine_snr_input, 'snr_in_cl': self.__determine_snr_input_class,
                                'snr_out': self.__determine_snr_output, 'snr_out_cl': self.__determine_snr_output_class,
                                'dsnr_all': self.__determine_dsnr_all, 'dsnr_cl': self.__determine_dsnr_class,
                                'ptq_loss': self.__determine_ptq_loss}

    def load_dataset(self, dataset: DatasetFromFile, noise_std: float=0.1, mode_train: int=0) -> None:
        """Loading the loaded dataset and transform it into right dataloader
        :param dataset:     Dataclass with dataset loaded from extern
        :param noise_std:   Adding noise standard deviation on input data
        :param mode_train:  Autoencoder Training Mode
                            [0: Autoencoder,
                            1: Denoising Autoencoder (mean),
                            2: Denoising Autoencoder (add random noise),
                            3: Denoising Autoencoder (add gaussian noise)]
        :return:            None
        """
        dataset0 = DatasetAutoencoder(
            dataset=dataset,
            noise_std=noise_std,
            mode_train=mode_train
        )
        self._prepare_dataset_for_training(
            data_set=dataset0,
            num_workers=0
        )

    def __do_training_epoch(self) -> float:
        """Do training during epoch of training
        Return:
            Floating value with training loss value
        """
        train_loss = 0.0
        total_batches = 0

        self._model.train(True)
        for tdata in self._train_loader[self._run_kfold]:
            data_x = tdata['in'].to(self._used_hw_dev)
            data_y = tdata['out'].to(self._used_hw_dev)
            data_p = self._model(data_x)[1]

            self._optimizer.zero_grad()
            if len(data_y) > 2:
                loss = self._loss_fn(flatten(data_p, 1), flatten(data_y, 1))
            else:
                loss = self._loss_fn(data_p, data_y)
            loss.backward()
            self._optimizer.step()

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
        self._total_batches_valid = 0
        valid_loss = 0.0

        self._model.eval()
        with inference_mode():
            for vdata in self._valid_loader[self._run_kfold]:
                data_x = vdata['in'].to(self._used_hw_dev)
                data_y = vdata['out'].to(self._used_hw_dev)
                data_m = vdata['mean'].to(self._used_hw_dev)
                data_id = vdata['class'].to(self._used_hw_dev)
                data_p = self._model(data_x)[1]

                self._total_batches_valid += 1
                if len(data_y) > 2:
                    valid_loss += self._loss_fn(flatten(data_p, 1), flatten(data_y, 1)).item()
                else:
                    valid_loss += self._loss_fn(data_p, data_y).item()

                # --- Calculating custom made metrics
                for metric_used in epoch_custom_metrics:
                    self._determine_epoch_metrics(metric_used)(data_x, data_p, data_m, metric=metric_used, id=data_id)

        return float(valid_loss / self._total_batches_valid)

    def __process_epoch_metrics_calculation(self, init_phase: bool, custom_made_metrics: list) -> None:
        """Function for preparing the custom-made metric calculation
        Args:
            init_phase:         Boolean decision if processing part is in init (True) or in training phase (False)
            custom_made_metrics:List with custom metrics for calculation during validation phase
        Return:
            None
        """
        assert check_keylist_elements_any(
            keylist=custom_made_metrics,
            elements=self.get_epoch_metric_custom_methods
        ), f"Used custom made metrics not found in: {self.get_epoch_metric_custom_methods} - Please adapt in settings!"
        # --- Init phase for generating empty data structure
        if init_phase:
            for key0 in custom_made_metrics:
                self.__metric_result.update({key0: list()})
                self.__metric_buffer.update({key0: list()})

        # --- Processing results
        else:
            for key0 in self.__metric_buffer.keys():
                self.__metric_result[key0].append(self.__metric_buffer[key0])
                self.__metric_buffer.update({key0: list()})

    def __determine_snr_input(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, **kwargs) -> None:
        out = calculate_snr_tensor(input_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[kwargs['metric']], list):
            self.__metric_buffer[kwargs['metric']] = out
        else:
            self.__metric_buffer[kwargs['metric']] = concatenate((self.__metric_buffer[kwargs['metric']], out), dim=0)

    def __determine_snr_input_class(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, **kwargs) -> None:
        out = self._separate_classes_from_label(
            pred=calculate_snr_tensor(input_waveform, mean_waveform),
            true=kwargs['id'], label=kwargs['metric']
        )
        if len(self.__metric_buffer[kwargs['metric']]) == 0:
            self.__metric_buffer[kwargs['metric']] = out[0]
        else:
            for idx, snr_class in enumerate(out[0]):
                old = self.__metric_buffer[kwargs['metric']][idx]
                self.__metric_buffer[kwargs['metric']][idx] = concatenate((old, snr_class), dim=0)

    def __determine_snr_output(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, **kwargs) -> None:
        out = calculate_snr_tensor(pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[kwargs['metric']], list):
            self.__metric_buffer[kwargs['metric']] = out
        else:
            self.__metric_buffer[kwargs['metric']] = concatenate((self.__metric_buffer[kwargs['metric']], out), dim=0)

    def __determine_snr_output_class(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, **kwargs) -> None:
        out = self._separate_classes_from_label(
            pred=calculate_snr_tensor(pred_waveform, mean_waveform),
            true=kwargs['id'], label=kwargs['metric']
        )
        if len(self.__metric_buffer[kwargs['metric']]) == 0:
            self.__metric_buffer[kwargs['metric']] = out[0]
        else:
            for idx, snr_class in enumerate(out[0]):
                old = self.__metric_buffer[kwargs['metric']][idx][0]
                self.__metric_buffer[kwargs['metric']][idx] = concatenate((old, snr_class), dim=0)

    def __determine_dsnr_all(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, **kwargs) -> None:
        out = calculate_dsnr_tensor(input_waveform, pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[kwargs['metric']], list):
            self.__metric_buffer[kwargs['metric']] = out
        else:
            self.__metric_buffer[kwargs['metric']] = concatenate((self.__metric_buffer[kwargs['metric']], out), dim=0)

    def __determine_dsnr_class(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, **kwargs) -> None:
        out = calculate_dsnr_tensor(input_waveform, pred_waveform, mean_waveform)
        if isinstance(self.__metric_buffer[kwargs['metric']], list):
            self.__metric_buffer[kwargs['metric']] = out
        else:
            self.__metric_buffer[kwargs['metric']] = concatenate((self.__metric_buffer[kwargs['metric']], out), dim=0)

    def __determine_ptq_loss(self, input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor, **kwargs) -> None:
        """if not hasattr(self.model, 'bit_config'):
            raise NotImplementedError('PTQ Test is only available with elasticAI.creator Models or '
                                      'model includes variable \"bit_config\" = [total_bitwidth, frac_bitwidth]')
        else:"""
        # --- Load model and make inference
        model_ptq = quantize_model_fxp(self._model, self._ptq_level[0], self._ptq_level[1])
        model_ptq.eval()
        pred_waveform_ptq = model_ptq(input_waveform)[1]

        # --- Calculate loss
        if len(input_waveform) > 2:
            loss = self._loss_fn(flatten(pred_waveform_ptq, 1), flatten(input_waveform, 1)).item() / self._total_batches_valid
        else:
            loss = self._loss_fn(pred_waveform_ptq, input_waveform).item() / self._total_batches_valid

        # --- Saving results
        if len(self.__metric_buffer[kwargs['metric']]):
            self.__metric_buffer[kwargs['metric']][0] = self.__metric_buffer[kwargs['metric']][0] + loss
        else:
            self.__metric_buffer[kwargs['metric']].append(loss)

    def do_training(self, path2save='', metrics=()) -> dict:
        """Start model training incl. validation and custom-own metric calculation
        Args:
            path2save:      Path for saving the results [Default: '' --> generate new folder]
            metrics:        List with strings of used metric [Default: empty]
        Returns:
            Dictionary with metrics from training (loss_train, loss_valid, own_metrics)
        """
        self._init_train(path2save=path2save, addon='_AE')
        if self._kfold_do:
            self._logger.info(f"Starting Kfold cross validation training in {self._settings_train.num_kfold} steps")

        metric_out = dict()
        path2model = str()
        path2model_init = join(self._path2save, f'model_ae_reset.pt')
        save(self._model.state_dict(), path2model_init)
        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S')
        self._logger.info(f'\nTraining starts on {timestamp_string}')
        self._logger.info("=====================================================================================")

        self.__process_epoch_metrics_calculation(True, metrics)
        for fold in np.arange(self._settings_train.num_kfold):
            # --- Init fold
            best_loss = np.array((1_000_000., 1_000_000.), dtype=float)
            patience_counter = self._settings_train.patience

            epoch_loss_train = list()
            epoch_loss_valid = list()

            self._model.load_state_dict(load(path2model_init, weights_only=False))
            self._run_kfold = fold
            if self._kfold_do:
                self._logger.info(f'\nStarting with Fold #{fold}')

            for epoch in range(0, self._settings_train.num_epochs):
                if self._settings_train.deterministic_do:
                    self._deterministic_generator.manual_seed(self._settings_train.deterministic_seed + epoch)

                loss_train = self.__do_training_epoch()
                loss_valid = self.__do_valid_epoch(metrics)
                epoch_loss_train.append(loss_train)
                epoch_loss_valid.append(loss_valid)
                self.__process_epoch_metrics_calculation(False, metrics)

                self._logger.info(f'... results of epoch {epoch + 1}/{self._settings_train.num_epochs} '
                                  f'[{(epoch + 1) / self._settings_train.num_epochs * 100:.2f} %]: '
                                  f'train_loss = {loss_train:.5f},'
                                  f'\tvalid_loss = {loss_valid:.5f},'
                                  f'\tdelta_loss = {loss_train - loss_valid:.6f}')

                # Tracking the best performance and saving the model
                if loss_valid < best_loss[1]:
                    best_loss = [loss_train, loss_valid]
                    path2model = join(self._path2temp, f'model_ae_fold{fold:03d}_epoch{epoch:04d}.pt')
                    save(self._model, path2model)
                    patience_counter = self._settings_train.patience
                else:
                    patience_counter -= 1

                # Early Stopping
                if patience_counter <= 0:
                    self._logger.info(f"... training stopped due to no change after {epoch + 1} epochs!")
                    break

            copy(path2model, self._path2save)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')

            # --- Saving results
            metric_fold = {
                'loss_train': epoch_loss_train,
                'loss_valid': epoch_loss_valid
            }
            metric_fold.update(self.__metric_result)
            metric_out.update({f"fold_{fold:03d}": metric_fold})

        # --- Ending of all trainings phases
        self._end_training_routine(timestamp_start)
        metric_save = self._converting_tensor_to_numpy(metric_out)
        np.save(f"{self._path2save}/metric_ae", metric_save, allow_pickle=True)
        return metric_out

    def do_post_training_validation(self, do_ptq: bool=False) -> dict:
        """Performing the post-training validation with the best model
        :param do_ptq:  Boolean for activating post training quantization during post-training validation
        :return:        Dictionary with model results
        """
        if cuda.is_available():
            cuda.empty_cache()

        # --- Do the Inference with Best Model
        overview_models = self.get_best_model('ae')
        if len(overview_models) == 0:
            raise RuntimeError(f"No models found on {self._path2save} - Please start training!")

        path2model = overview_models[0]
        if do_ptq:
            model_test = quantize_model_fxp(
                model=load(path2model, weights_only=False),
                total_bits=self._ptq_level[0],
                frac_bits=self._ptq_level[1]
            )
        else:
            model_test = load(path2model, weights_only=False)

        self._logger.info("=================================================================")
        self._logger.info(f"Do Validation with best model: {path2model}")

        pred_model = randn(32, 1)
        feat_model = randn(32, 1)
        clus_orig_list = randn(32, 1)
        data_orig_list = randn(32, 1)

        first_cycle = True
        model_test.eval()
        for vdata in self._valid_loader[-1]:
            feat, pred = model_test(vdata['in'].to(self._used_hw_dev))
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
        return self._getting_data_for_plotting(
            data_orig_list.numpy(),
            clus_orig_list.numpy(),
            {'feat': result_feat, 'pred': result_pred},
            addon='ae'
        )
