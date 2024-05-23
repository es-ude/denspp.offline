import numpy as np
from os.path import join
from shutil import copy
from datetime import datetime
from torch import Tensor, load, save, from_numpy
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset, training_pytorch


class train_nn(training_pytorch):
    """Class for Handling the Training of Classifiers within Recurrent Neural Networks"""
    def __init__(self, config_train: Config_PyTorch, config_data: Config_Dataset, do_train=True) -> None:
        training_pytorch.__init__(self, config_train, config_data, do_train)

    def __do_training_epoch(self) -> [float, float]:
        """Do training during epoch of training"""
        train_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0

        self.model.train(True)
        for tdata in self.train_loader[self._run_kfold]:
            self.optimizer.zero_grad()
            pred_con, pred_cl = self.model(tdata['in'].to(self.used_hw_dev))

            loss = self.loss_fn(pred_con, tdata['out'].to(self.used_hw_dev))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total_batches += 1
            total_correct += int(sum(pred_cl == tdata['out']))
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
            pred_cl, dec_cl = self.model(vdata['in'].to(self.used_hw_dev))

            valid_loss += self.loss_fn(pred_cl, vdata['out'].to(self.used_hw_dev)).item()
            total_batches += 1
            total_correct += int(sum(dec_cl == vdata['out']))
            total_samples += len(vdata['in'])

        valid_acc = total_correct / total_samples
        valid_loss = valid_loss / total_batches

        return valid_loss, valid_acc

    def do_training(self, path2save='') -> list:
        """Start model training incl. validation and custom-own metric calculation"""
        self._init_train(path2save)
        self._save_config_txt('_rnn')

        # --- Handling Kfold cross validation training
        if self._do_kfold:
            print(f"Starting Kfold cross validation training in {self.settings.num_kfold} steps")

        metrics_own = list()
        path2model = str()
        path2model_init = join(self._path2save, f'model_rnn_reset.pth')
        save(self.model.state_dict(), path2model_init)
        timestamp_start = datetime.now()
        timestamp_string = timestamp_start.strftime('%H:%M:%S')
        print(f'\nTraining starts on {timestamp_string}'
              f"\n=====================================================================================")

        for fold in np.arange(self.settings.num_kfold):
            best_loss = [1e6, 1e6]
            best_acc = [0.0, 0.0]
            # Init fold
            epoch_metric = list()
            self.model.load_state_dict(load(path2model_init))
            self._run_kfold = fold
            self._init_writer()

            if self._do_kfold:
                print(f'\nStarting with Fold #{fold}')

            for epoch in range(0, self.settings.num_epochs):
                train_loss, train_acc = self.__do_training_epoch()
                valid_loss, valid_acc = self.__do_valid_epoch()

                print(f'... results of epoch {epoch + 1}/{self.settings.num_epochs} '
                      f'[{(epoch + 1) / self.settings.num_epochs * 100:.2f} %]: '
                      f'train_loss = {train_loss:.5f}, train_acc = {100 * train_acc:.2f} % - '
                      f'valid_loss = {valid_loss:.5f}, valid_acc = {100 * valid_acc:.2f} %')

                # Log the running loss averaged per batch for both training and validation
                self._writer.add_scalar('Loss_train (CL)', train_loss, epoch+1)
                self._writer.add_scalar('Loss_valid (CL)', valid_loss, epoch+1)
                self._writer.add_scalar('Acc_train (CL)', train_acc, epoch+1)
                self._writer.add_scalar('Acc_valid (CL)', valid_acc, epoch+1)
                self._writer.flush()

                # Tracking the best performance and saving the model
                if valid_loss < best_loss[1]:
                    best_loss = [train_loss, valid_loss]
                    best_acc = [train_acc, valid_acc]
                    path2model = join(self._path2temp, f'model_rnn_fold{fold:03d}_epoch{epoch:04d}.pth')
                    save(self.model, path2model)

                # Saving metrics after each epoch
                epoch_metric.append(np.array((train_acc, valid_acc), dtype=float))

            # --- Saving metrics after each fold
            metrics_own.append(epoch_metric)
            copy(path2model, self._path2save)
            self._save_train_results(best_loss[0], best_loss[1], 'Loss')
            self._save_train_results(best_acc[0], best_acc[1], 'Acc.')

        # --- Ending of all trainings phases
        self._end_training_routine(timestamp_start)

        return metrics_own

    def do_validation_after_training(self, num_output: int) -> dict:
        """Performing the validation with the best model after training"""
        # --- Getting data from validation set for inference
        data_train = self.get_data_points(num_output, use_train_dataloader=True)
        data_valid = self.get_data_points(num_output, use_train_dataloader=False)

        # --- Do the Inference with Best Model
        print(f"\nDoing the inference with validation data on best model")
        model_inference = load(self.get_best_model('rnn')[0])
        if not isinstance(data_valid['in'], Tensor):
            data_train = from_numpy(data_train['out'])
            data_input = from_numpy(data_valid['in'])
            data_output = from_numpy(data_valid['out'])
        else:
            data_train = data_train['out']
            data_input = data_valid['in']
            data_output = data_valid['out']

        yclus = model_inference(data_input)[1]
        yclus = yclus.detach().numpy()

        # --- Producing the output
        output = dict()
        output.update({'settings': self.settings, 'date': datetime.now().strftime('%d/%m/%Y, %H:%M:%S')})
        output.update({'train_clus': data_train, 'valid_clus': data_output})
        output.update({'input': data_input, 'yclus': yclus})
        output.update({'cl_dict': self.cell_classes})

        # --- Saving dict
        np.save(join(self.get_saving_path(), 'results.mat'), output)
        return output
