import numpy as np
from os.path import join
from logging import getLogger, Logger
from shutil import copy
from datetime import datetime
from torch import Tensor, zeros, load, save, concatenate, inference_mode, cuda, cat, randn, add, div

from denspp.offline import check_keylist_elements_any
from denspp.offline.dnn import SettingsPytorch, DatasetFromFile
from denspp.offline.dnn.ptq_help import quantize_model_fxp
from denspp.offline.dnn.training.common_train import SettingsDataset, PyTorchHandler
from denspp.offline.dnn.training.classifier_dataset import DatasetClassifier, DatasetAutoencoderClassifier
from denspp.offline.metric.data_torch import (
    calculate_number_true_predictions,
    calculate_precision,
    calculate_recall,
    calculate_fbeta
)


class TrainClassifier(PyTorchHandler):
    _logger: Logger

    def __init__(self, config_train: SettingsPytorch, config_data: SettingsDataset, do_train: bool=True) -> None:
        """Class for Handling Training of Classifiers
        :param config_data:     Settings for handling and loading the dataset (just for saving)
        :param config_train:    Settings for handling the PyTorch Trainings Routine
        :param do_train:        Do training of model otherwise only inference
        :return:                None
        """
        PyTorchHandler.__init__(self, config_train, config_data, do_train)
        self._logger = getLogger(__name__)
        self.__metric_buffer = dict()
        self.__metric_result = dict()
        self._metric_methods = {'accuracy': self.__determine_accuracy_per_class,
                                'precision': self.__determine_buffering_metric_calculation,
                                'recall': self.__determine_buffering_metric_calculation,
                                'fbeta': self.__determine_buffering_metric_calculation,
                                'ptq_loss': self.__determine_ptq_loss}

    def load_dataset(self, dataset: DatasetFromFile) -> None:
        """Loading the loaded dataset and transform it into right dataloader
        :param dataset:     Dataclass with dataset loaded from extern
        :return:            None
        """
        #TODO: Wie enable ich die Autoencoder-Classifier Geschichte?
        dataset0 = DatasetClassifier(
            dataset=dataset,
        )
        self._prepare_dataset_for_training(
            data_set=dataset0,
            num_workers=0
        )

    def __do_training_epoch(self) -> tuple[float, float]:
        """Do training during epoch of training
        Return:
            Floating value of training loss and accuracy of used epoch
        """
        train_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0

        self._model.train(True)
        for tdata in self._train_loader[self._run_kfold]:
            self._optimizer.zero_grad()
            tdata_out = tdata['out'].to(self._used_hw_dev)
            pred_cl, dec_cl = self._model(tdata['in'].to(self._used_hw_dev))

            loss = self._loss_fn(pred_cl, tdata_out)
            loss.backward()
            self._optimizer.step()

            train_loss += loss.item()
            total_batches += 1
            total_correct += calculate_number_true_predictions(dec_cl, tdata_out)
            total_samples += len(tdata['in'])

        train_acc = float(int(total_correct) / total_samples)
        train_loss = float(train_loss / total_batches)
        return train_loss, train_acc

    def __do_valid_epoch(self, epoch_custom_metrics: list) -> tuple[float, float]:
        """Do validation during epoch of training
        Args:
            epoch_custom_metrics:   List with entries of custom-made metric calculations
        Return:
            Floating value of validation loss and validation accuracy of used epoch
        """
        valid_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0

        self._model.eval()
        with inference_mode():
            for vdata in self._valid_loader[self._run_kfold]:
                # --- Validation phase of model
                pred_cl, dec_cl = self._model(vdata['in'].to(self._used_hw_dev))
                true_cl = vdata['out'].to(self._used_hw_dev)

                valid_loss += self._loss_fn(pred_cl, true_cl).item()
                total_batches += 1
                total_correct += calculate_number_true_predictions(dec_cl, true_cl)
                total_samples += len(vdata['in'])

                # --- Calculating custom made metrics
                for metric_used in epoch_custom_metrics:
                    self._determine_epoch_metrics(metric_used)(dec_cl, true_cl, metric=metric_used, frame=vdata['in'].to(self._used_hw_dev))

        valid_acc = float(int(total_correct) / total_samples)
        valid_loss = float(valid_loss / total_batches)
        return valid_loss, valid_acc

    def __process_epoch_metrics_calculation(self, init_phase: bool, custom_made_metrics: list) -> None:
        """Function for preparing the custom-made metric calculation
        Args:
            init_phase:         Boolean decision if processing part is in init (True) or in post-training phase (False)
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
                match key0:
                    case 'accuracy':
                        self.__metric_buffer.update(
                            {key0: [zeros((len(self._cell_classes),)), zeros((len(self._cell_classes),))]}
                        )
                    case 'precision':
                        self.__metric_buffer.update({key0: [[], []]})
                    case 'recall':
                        self.__metric_buffer.update({key0: [[], []]})
                    case 'fbeta':
                        self.__metric_buffer.update({key0: [[], []]})
                    case 'ptq_loss':
                        self.__metric_buffer.update({key0: [zeros((1,)), zeros((1,))]})
        # --- Processing results
        else:
            for key0 in self.__metric_buffer.keys():
                match key0:
                    case 'accuracy':
                        self.__metric_result[key0].append(div(self.__metric_buffer[key0][0], self.__metric_buffer[key0][1]))
                        self.__metric_buffer.update({key0: [zeros((len(self._cell_classes),)), zeros((len(self._cell_classes),))]})
                    case 'precision':
                        out = self._separate_classes_from_label(
                            self.__metric_buffer[key0][0], self.__metric_buffer[key0][1], key0,
                            calculate_precision
                        )
                        self.__metric_result[key0].append(out[0])
                        self.__metric_buffer.update({key0: [[], []]})
                    case 'recall':
                        out = self._separate_classes_from_label(
                            self.__metric_buffer[key0][0], self.__metric_buffer[key0][1], key0,
                            calculate_recall
                        )
                        self.__metric_result[key0].append(out[0])
                        self.__metric_buffer.update({key0: [[], []]})
                    case 'fbeta':
                        out = self._separate_classes_from_label(
                            self.__metric_buffer[key0][0], self.__metric_buffer[key0][1], key0,
                            calculate_fbeta
                        )
                        self.__metric_result[key0].append(out[0])
                        self.__metric_buffer.update({key0: [[], []]})
                    case 'ptq_loss':
                        self.__metric_result[key0].append(div(self.__metric_buffer[key0][0], self.__metric_buffer[key0][1]))
                        self.__metric_buffer.update({key0: [zeros((1,)), zeros((1,))]})

    def __determine_accuracy_per_class(self, pred: Tensor, true: Tensor, **kwargs) -> None:
        out = self._separate_classes_from_label(pred, true, kwargs['metric'], calculate_number_true_predictions)
        self.__metric_buffer[kwargs['metric']][0] = add(self.__metric_buffer[kwargs['metric']][0], out[0])
        self.__metric_buffer[kwargs['metric']][1] = add(self.__metric_buffer[kwargs['metric']][1], out[1])

    def __determine_buffering_metric_calculation(self, pred: Tensor, true: Tensor, **kwargs) -> None:
        if len(self.__metric_buffer[kwargs['metric']][0]) == 0:
            self.__metric_buffer[kwargs['metric']][0] = true
            self.__metric_buffer[kwargs['metric']][1] = pred
        else:
            self.__metric_buffer[kwargs['metric']][0] = concatenate((self.__metric_buffer[kwargs['metric']][0], true), dim=0)
            self.__metric_buffer[kwargs['metric']][1] = concatenate((self.__metric_buffer[kwargs['metric']][1], pred), dim=0)

    def __determine_ptq_loss(self, pred: Tensor, true: Tensor, **kwargs) -> None:
        model_ptq = quantize_model_fxp(
            model=self._model,
            total_bits=self._ptq_level[0],
            frac_bits=self._ptq_level[1]
        )
        model_ptq.eval()
        pred_cl, dec_cl = model_ptq(kwargs['frame'])
        num_true = calculate_number_true_predictions(dec_cl, true)

        self.__metric_buffer[kwargs['metric']][0] = add(self.__metric_buffer[kwargs['metric']][0], num_true)
        self.__metric_buffer[kwargs['metric']][1] = add(self.__metric_buffer[kwargs['metric']][1], kwargs['frame'].shape[0])

    def do_training(self, path2save: str='', metrics: list=()) -> dict:
        """Start model training incl. validation and custom-own metric calculation
        Args:
            path2save:      Path for saving the results [Default: '' --> generate new folder]
            metrics:        List with strings of used metric ['acc'] [Default: empty]
        Returns:
            Dictionary with metrics from training (loss_train, loss_valid, own_metrics)
        """
        self._init_train(path2save=path2save, addon='_CL')
        if self._kfold_do:
            self._logger.info(f"Starting Kfold cross validation training in {self._settings_train.num_kfold} steps")

        path2model = str()
        path2model_init = join(self._path2save, f'model_class_reset.pt')
        save(self._model.state_dict(), path2model_init)
        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S')
        self._logger.info(f'Training starts on {timestamp_string}')
        self._logger.info(f"=====================================================================================")

        metric_out = dict()
        self.__process_epoch_metrics_calculation(True, metrics)
        for fold in np.arange(self._settings_train.num_kfold):
            # --- Init fold
            best_loss = [1e6, 1e6]
            best_acc = [0.0, 0.0]
            patience_counter = self._settings_train.patience

            epoch_train_acc = list()
            epoch_valid_acc = list()
            epoch_train_loss = list()
            epoch_valid_loss = list()

            self._model.load_state_dict(load(path2model_init, weights_only=False))
            self._run_kfold = fold

            if self._kfold_do:
                self._logger.info(f'Starting with Fold #{fold}')

            for epoch in range(0, self._settings_train.num_epochs):
                if self._settings_train.deterministic_do:
                    self._deterministic_generator.manual_seed(self._settings_train.deterministic_seed + epoch)

                train_loss, train_acc = self.__do_training_epoch()
                valid_loss, valid_acc = self.__do_valid_epoch(metrics)
                self._logger.info(f'... results of epoch {epoch + 1}/{self._settings_train.num_epochs} '
                      f'[{(epoch + 1) / self._settings_train.num_epochs * 100:.2f} %]: '
                      f'train_loss = {train_loss:.5f}, delta_loss = {train_loss-valid_loss:.5f}, '
                      f'train_acc = {100* train_acc:.4f} %, delta_acc = {100 * (train_acc-valid_acc):.4f} %')

                # Saving metrics after each epoch
                epoch_train_acc.append(train_acc)
                epoch_train_loss.append(train_loss)
                epoch_valid_acc.append(valid_acc)
                epoch_valid_loss.append(valid_loss)
                self.__process_epoch_metrics_calculation(False, metrics)

                # Tracking the best performance and saving the model
                if valid_loss < best_loss[1]:
                    best_loss = [train_loss, valid_loss]
                    best_acc = [train_acc, valid_acc]
                    path2model = join(self._path2temp, f'model_class_fold{fold:03d}_epoch{epoch:04d}.pt')
                    save(self._model, path2model)
                    patience_counter = self._settings_train.patience
                else:
                    patience_counter -= 1

                # Early Stopping
                if patience_counter <= 0:
                    self._logger.info(f"... training stopped due to no change after {epoch+1} epochs!")
                    break

            copy(path2model, self._path2save)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')
            self._save_train_results(best_acc[0], best_acc[1], 'Acc.')

            # --- Saving metrics after each fold
            metric_fold = {
                "acc_train": epoch_train_acc,
                "acc_valid": epoch_valid_acc,
                "loss_train": epoch_train_loss,
                "loss_valid": epoch_valid_loss
            }
            metric_fold.update(self.__metric_result)
            metric_out.update({f"fold_{fold:03d}": metric_fold})

        # --- Ending of all trainings phases
        self._end_training_routine(timestamp_start)
        metric_save = self._converting_tensor_to_numpy(metric_out)
        np.save(f"{self._path2save}/metric_cl", metric_save, allow_pickle=True)
        return metric_save

    def do_post_training_validation(self, do_ptq: bool=False) -> dict:
        """Performing the post-training validation with the best model
        :param do_ptq:  Boolean for activating post training quantization during post-training validation
        :return:        Dictionary with model results
        """
        if cuda.is_available():
            cuda.empty_cache()

        # --- Do the Inference with Best Model

        overview_models = self.get_best_model('cl')
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

        clus_pred_list = randn(32, 1)
        clus_orig_list = randn(32, 1)
        data_orig_list = randn(32, 1)

        first_cycle = True
        model_test.eval()
        for vdata in self._valid_loader[-1]:
            _, clus_pred = model_test(vdata['in'].to(self._used_hw_dev))
            if first_cycle:
                clus_pred_list = clus_pred.detach().cpu()
                clus_orig_list = vdata['out']
                data_orig_list = vdata['in']
            else:
                clus_pred_list = cat((clus_pred_list, clus_pred.detach().cpu()), dim=0)
                clus_orig_list = cat((clus_orig_list, vdata['out']), dim=0)
                data_orig_list = cat((data_orig_list, vdata['in']), dim=0)
            first_cycle = False

        # --- Preparing output
        result_pred = clus_pred_list.numpy()
        return self._getting_data_for_plotting(
            valid_input=data_orig_list.numpy(),
            valid_label=clus_orig_list.numpy(),
            results={'yclus': result_pred},
            addon='cl'
        )
