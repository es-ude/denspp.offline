import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger, Logger
from typing import Any
from pathlib import Path

from denspp.offline import (
    get_path_to_project,
    check_keylist_elements_all
)
from denspp.offline.dnn import (
    DatasetFromFile,
    SettingsDataset,
    DefaultSettingsDataset
)
from denspp.offline.dnn.training import (
    TrainClassifier,
    SettingsClassifier,
    DefaultSettingsTrainingCE,
    TrainAutoencoder,
    SettingsAutoencoder,
    DefaultSettingsTrainingMSE
)
from denspp.offline.logger import define_logger_runtime
from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.dnn.model_library import DatasetLoaderLibrary
import denspp.offline.dnn.plots as dnn_plot


@dataclass
class TrainingResults:
    """Dataclass with returning results from training routine of a deep learning model
    Attributes:
        metrics:        Dictionary with metrics from training
        data:           Dictionary with data from post-training validation
        path:           Path to training results
        metrics_custom: List with names of all custom metrics calculated during training
    """
    metrics: dict
    data: dict
    path: Path
    metrics_custom: list[str]

@dataclass
class SettingsMLPipeline:
    """Configuration class for handling the training phase of deep neural networks
    Attributes:
        mode_train:         Integer of selected training routine regarding the training handler [0: Classifier, 1: Autoencoder]
        do_block:           Boolean value to block the generated plots after training
        do_ptq:             Apply Post Training Quantization Scheme during Training
        ptq_total_bitwidth: Integer for total bitwidth in PTQ
        ptq_frac_bitwidth:  Integer for fractional bitwidth in PTQ
    """
    mode_train: int
    do_block: bool
    do_ptq: bool
    ptq_total_bitwidth: int
    ptq_frac_bitwidth: int


DefaultSettingsMLPipeline = SettingsMLPipeline(
    mode_train=0,
    do_block=True,
    do_ptq=False,
    ptq_total_bitwidth=8,
    ptq_frac_bitwidth=4,
)


class PyTorchPlot:
    _logger: Logger

    def __init__(self):
        """Class for handling all suitable plot options for the PyTorch Training Handler"""
        self._logger: Logger = getLogger(__name__)

    @staticmethod
    def performance_autoencoder_mnist(data: TrainingResults, show_plot: bool=False) -> None:
        """Plotting the dataset content with initial and predicted values after autoencoder training
        :param data:        Dataclass TrainingResults with results from Training
        :param show_plot:   Boolean value to show the plot
        :return:            None
        """
        dnn_plot.plot_mnist_dataset(
            data=data.data['input'],
            label=data.data['valid_clus'],
            title="_input",
            path2save=str(data.path),
            show_plot=False
        )
        dnn_plot.plot_mnist_dataset(
            data=data.data['pred'],
            label=data.data['valid_clus'],
            title="_predicted",
            path2save=str(data.path),
            show_plot=show_plot
        )

    @staticmethod
    def loss(data: TrainingResults, loss_type: str, fold_num: int=0, epoch_zoom=None, show_plot:bool=False) -> None:
        """Plotting the loss values of each epoch during training
        :param data:        Dataclass TrainingResults with results from Training
        :param loss_type:   String with name of the used loss function
        :param fold_num:    Integer with fold number to analyse
        :param epoch_zoom:  Optional list with ranges for zooming loss data
        :param show_plot:   Boolean value to show the plot
        :return:            None
        """
        fold_overview = [key for key in data.metrics.keys()]
        if len(fold_overview) == 0:
            raise AttributeError("No fold available in dataset")
        else:
            used_fold = fold_overview[fold_num]
            dnn_plot.plot_loss(
                loss_train=data.metrics[used_fold]['loss_train'],
                loss_valid=data.metrics[used_fold]['loss_valid'],
                type=loss_type,
                path2save=str(data.path),
                epoch_zoom=epoch_zoom,
                show_plot=show_plot
            )

    @staticmethod
    def custom_loss(data: TrainingResults, fold_num: int=0, epoch_zoom=None, show_plot:bool=False) -> None:
        """Plotting the custom metrics of each epoch during training
        :param data:        Dataclass TrainingResults with results from Training
        :param fold_num:    Integer with fold number to analyse
        :param epoch_zoom:  Optional list with ranges for zooming loss data
        :param show_plot:   Boolean value to show the plot
        :return:            None
        """
        # TODO: Adapt function for custom losses
        # TODO: Add logy plot option to loss function
        # TODO: Add cluster metric value to feat. space plots
        fold_overview = [key for key in data.metrics.keys()]
        if len(fold_overview) == 0:
            raise AttributeError("No fold available in dataset")

        used_fold = fold_overview[fold_num]
        for metric_check in data.metrics_custom:
            if metric_check in list(data.metrics[used_fold].keys()):
                dnn_plot.plot_custom_loss(
                    data=data.metrics[used_fold][metric_check],
                    loss_name=metric_check,
                    do_boxplot=False,
                    epoch_zoom=epoch_zoom,
                    path2save=str(data.path),
                    show_plot=show_plot
                )

    @staticmethod
    def statistics(data: TrainingResults, show_plot: bool=False) -> None:
        """Plotting the statistics of the used dataset for training
        :param data:        Dataclass TrainingResults with results from Training
        :param show_plot:   Boolean value to show the plot
        :return:            None
        """
        dnn_plot.plot_statistic(
            train_cl=data.data['train_clus'],
            valid_cl=data.data['valid_clus'],
            path2save=str(data.path),
            cl_dict=data.data['cl_dict'],
            show_plot=show_plot
        )

    @staticmethod
    def performance_classifier(data: TrainingResults, fold_num: int=0, epoch_zoom=None, show_plot: bool=False) -> None:
        """Plotting the results after classifier training
        :param data:        Dataclass TrainingResults with results from Training
        :param fold_num:    Integer with fold number to analyse
        :param epoch_zoom:  Optional list with ranges for zooming loss data
        :param show_plot:   Boolean value to show the plot
        :return:            None
        """
        overview_fold = [key for key in data.metrics.keys()]
        if len(overview_fold) == 0:
            raise AttributeError("No fold available in dataset")

        used_fold = overview_fold[fold_num]
        dnn_plot.plot_loss(
            loss_train=data.metrics[used_fold]['acc_train'],
            loss_valid=data.metrics[used_fold]['acc_valid'],
            type='Accuracy',
            epoch_zoom=epoch_zoom,
            path2save=str(data.path),
            show_plot=False
        )
        dnn_plot.plot_confusion(
            pred_labels=data.data['valid_clus'],
            true_labels=data.data['yclus'],
            path2save=str(data.path),
            cl_dict=data.data['cl_dict'],
            show_plots=show_plot
        )

    @staticmethod
    def performance_autoencoder(data: TrainingResults, mean_value: np.ndarray, show_plot: bool=False) -> None:
        """Plotting the results after autoencoder training
        :param data:        Dataclass TrainingResults with results from Training
        :param mean_value:  Numpy array with mean waveforms for each class
        :param show_plot:   Boolean value to show the plot
        :return:            None
        """
        dnn_plot.results_autoencoder_training(
            cl_dict=data.data['cl_dict'],
            feat=data.data['feat'],
            yin=data.data['input'],
            ypred=data.data['pred'],
            ymean=mean_value,
            yclus=data.data['valid_clus'],
            path=str(data.path),
            show_plot=show_plot
        )


class PyTorchTrainer:
    _logger: Logger
    _plotter: PyTorchPlot
    _settings_ml: SettingsMLPipeline
    _settings_data: SettingsDataset
    _settings_model: SettingsClassifier | SettingsAutoencoder
    _path2config: Path
    _dataloader: Any
    _conf_available: bool

    def __init__(self, usecase_name: str, default_trainer: int,
                 default_model: str= '', path2config: str= 'config', generate_configs: bool=True) -> None:
        """Class for handling and wrapping all PyTorch Training Routines incl. Report Generation and Plotting
        :param usecase_name:        String with name of use-case
        :param default_trainer:     Integer with default training routine [0: Classifier, 1: Autoencoder]
        :param default_model:       String with name of default model
        :param path2config:         Path to folder with configuration files
        :param generate_configs:    Boolean for generating configuration files if not there
        :return:                    None
        """
        define_logger_runtime(save_file=False)
        self._logger: Logger = getLogger(__name__)
        self._plotter = PyTorchPlot()
        self._path2config = Path(get_path_to_project(path2config))
        self._settings_ml = self._get_config_ml(
            usecase_name=usecase_name,
            default_training_mode=default_trainer
        )
        if generate_configs:
            self.__prepare_training(
                usecase_name=usecase_name,
                default_trainer=default_trainer,
                default_model=default_model
            )

    def __prepare_training(self, usecase_name: str, default_model: str, default_trainer: int) -> None:
        self._path2config.mkdir(parents=True, exist_ok=True)
        self._conf_available = self.config_available

        self._dataloader = self._get_dataset_loader()
        self._settings_data = self._get_config_dataset(
            default_dataset_name=usecase_name,
            usecase_name=usecase_name
        )
        match self._settings_ml.mode_train:
            case 0:
                self._settings_model = self._get_config_classifier(
                    default_model_name=default_model,
                    usecase_name=usecase_name
                )
            case 1:
                self._settings_model = self._get_config_autoencoder(
                    default_model_name=default_model,
                    usecase_name=usecase_name
                )
            case _:
                raise NotImplementedError("Training routine not implemented")

    @property
    def config_available(self) -> bool:
        """Checking if configs are in the folder available or must be initialized"""
        return self._path2config.exists() and len(list(self._path2config.glob("Config*_*.yaml"))) > 0

    @property
    def path2config(self) -> Path:
        """Returning the absolute path to config folder"""
        return self._path2config.absolute()

    @property
    def get_custom_metric_calculation(self) -> list[str]:
        """Returning an overview of custom metric calculation methods during PyTorch Training"""
        match self._settings_ml.mode_train:
            case 0:
                method = TrainClassifier
                default = DefaultSettingsTrainingCE
            case 1:
                method = TrainAutoencoder
                default = DefaultSettingsTrainingMSE
            case _:
                raise NotImplementedError("Training routine not implemented")
        return method(
            config_train=default,
            config_data=DefaultSettingsDataset,
            do_train=False
        ).get_epoch_metric_custom_methods

    def _get_config_ml(self, usecase_name: str, default_training_mode: int=0) -> SettingsMLPipeline:
        default_set = deepcopy(DefaultSettingsMLPipeline)
        default_set.mode_train = default_training_mode
        return YamlHandler(
            template=default_set,
            path=str(self.path2config),
            file_name=f'ConfigTraining_{usecase_name}'
        ).get_class(SettingsMLPipeline)

    def _get_config_dataset(self, default_dataset_name: str, usecase_name: str) -> SettingsDataset:
        default_set: SettingsMLPipeline = deepcopy(DefaultSettingsDataset)
        default_set.data_type = default_dataset_name
        return YamlHandler(
            template=default_set,
            path=str(self.path2config),
            file_name=f'ConfigDataset_{usecase_name}'
        ).get_class(SettingsDataset)

    @staticmethod
    def _get_dataset_loader() -> Any:
        datalib = DatasetLoaderLibrary().get_registry()
        matches = [item for item in datalib.get_library_overview() if 'DatasetLoader' == item]
        if len(matches) == 0:
            raise AttributeError("No DatasetLoader available")
        return datalib.build_object(matches[0])

    def get_dataset(self) -> DatasetFromFile:
        """Getting the dataset with rawdata, label and label names for training a deep learning model
        :return:    Dataclass DatasetFromFile with loaded dataset
        """
        return self._dataloader(self._settings_data).load_dataset()

    def _get_config_classifier(self, default_model_name: str, usecase_name: str) -> SettingsClassifier:
        default_set: SettingsClassifier = deepcopy(DefaultSettingsTrainingCE)
        default_set.model_name = default_model_name
        return YamlHandler(
            template=default_set,
            path=str(self.path2config),
            file_name=f'ConfigClassifier_{usecase_name}'
        ).get_class(SettingsClassifier)

    def _prepare_training_classifier(self) -> TrainClassifier:
        """PyTorch Training Routing for Classifiers
        :return:            Training Handler
        """
        # --- Processing Step #0: Get dataset and build model
        used_dataset = self.get_dataset()
        model_signature = self._settings_model.get_signature()
        if len(model_signature) and check_keylist_elements_all(
                keylist=model_signature,
                elements=['input_size', 'output_size']
        ):
            sets = dict(
                input_size=int(np.prod(used_dataset.data.shape[1:])),
                output_size=np.unique(used_dataset.label).size
            )
        else:
            sets = dict()
        used_model = self.get_model(**sets)

        # ---Processing Step #1: Prepare Training Handler
        train_handler = TrainClassifier(
            config_train=self._settings_model,
            config_data=self._settings_data,
            do_train=True
        )
        train_handler.load_model(model=used_model)
        train_handler.load_dataset(dataset=used_dataset)
        return train_handler

    def _get_config_autoencoder(self, default_model_name: str, usecase_name: str) -> SettingsAutoencoder:
        default_set: SettingsAutoencoder = deepcopy(DefaultSettingsTrainingMSE)
        default_set.model_name = default_model_name
        return YamlHandler(
            template=default_set,
            path=str(self.path2config),
            file_name=f'ConfigAutoencoder_{usecase_name}'
        ).get_class(SettingsAutoencoder)

    def _prepare_training_autoencoder(self) -> TrainAutoencoder:
        """PyTorch Training Routing for Autoencoders
        :return:            Training handler
        """
        # --- Processing Step #0: Get dataset and build model
        used_dataset = self.get_dataset()
        model_signature = self._settings_model.get_signature()
        if len(model_signature) and check_keylist_elements_all(
                keylist=model_signature,
                elements=['input_size', 'output_size']
        ):
            if self._settings_model.feat_size:
                sets = dict(
                    input_size=int(np.prod(used_dataset.data.shape[1:])),
                    output_size=self._settings_model.feat_size
                )
            else:
                sets = dict(
                    input_size=int(np.prod(used_dataset.data.shape[1:])),
                    output_size=int(np.prod(used_dataset.data.shape[1:])),
                )
        else:
            sets = dict()
        used_model = self.get_model(**sets)

        # ---Processing Step #1: Prepare Trainings Handler
        train_handler = TrainAutoencoder(
            config_train=self._settings_model,
            config_data=self._settings_data,
            do_train=True
        )
        train_handler.load_model(model=used_model)
        train_handler.load_dataset(dataset=used_dataset)
        return train_handler

    def get_model(self, *args, **kwargs):
        """Returning the deep learning model for training loaded from ModelLibrary"""
        return self._settings_model.get_model(*args, **kwargs)

    def _run_training(self, path2save=Path(".")) -> TrainingResults:
        # --- Processing Step #1: Prepare Trainings handler with dataset and model
        match self._settings_ml.mode_train:
            case 0:
                dut: TrainClassifier = self._prepare_training_classifier()
                addon = 'cl'
            case 1:
                dut: TrainAutoencoder = self._prepare_training_autoencoder()
                addon = 'ae'
            case _:
                raise NotImplementedError

        # --- Processing Step #2: Do Training and Validation
        dut.define_ptq_level(
            total_bitwidth=self._settings_ml.ptq_total_bitwidth,
            frac_bitwidth=self._settings_ml.ptq_frac_bitwidth
        )
        metrics = dut.do_training(path2save=path2save)
        data_result = dut.do_post_training_validation(do_ptq=self._settings_ml.do_ptq)

        # --- Processing Step #3: Saving results
        results = TrainingResults(
            metrics=metrics,
            data=data_result,
            path=dut.get_saving_path(),
            metrics_custom=dut.get_epoch_metric_custom_methods
        )
        self._save_results(
            data=results,
            path2save=dut.get_saving_path(),
            addon=addon
        )
        return results

    def _save_results(self, data: TrainingResults, path2save: Path, addon: str) -> None:
        data2save = path2save / f'results_{addon}.npy'
        self._logger.debug(f"... saving results: {data2save}")
        np.save(data2save, data, allow_pickle=True)

    def do_training(self, path2save=Path(".")) -> TrainingResults:
        """Running PyTorch Training for specified configuration
        :param path2save:   Path to save the results and models after training [default runs/<YYYYMMDD>_<model>]
        :return:            Dataclass TrainingResults with internal metrics, data and path to run folder
        """
        if not self._conf_available:
            raise AttributeError("Configs are generated - Please adapt and restart!")
        results = self._run_training(path2save=path2save)
        return results

    def do_plot_dataset(self, path2save: Path=Path(".")) -> None:
        """Function for plotting the dataset content
        :param path2save:   Path to save the dataset plot
        :return:            None
        """
        dataset = self.get_dataset()
        match self._settings_data.data_type.lower():
            case "mnist":
                dnn_plot.plot_mnist_dataset(
                    data=dataset.data,
                    label=dataset.label,
                    title='',
                    path2save=str(path2save.absolute()),
                    show_plot=False
                )
            case "waveforms":
                dnn_plot.plot_waveforms_dataset(
                    dataset=dataset,
                    num_samples_class=10,
                    path2save=str(path2save.absolute()),
                    show_plot=False
                )
            case _:
                dnn_plot.plot_frames_dataset(
                    data=dataset,
                    take_samples=100,
                    do_norm=True,
                    add_subtitle=False,
                    path2save=str(path2save.absolute()),
                    show_plot=False
                )

    def do_plot_results(self, results: TrainingResults, epoch_zoom=None) -> None:
        """Function for plotting the results from training [metric, performance, data statistics]
        :param results:     Dataclass TrainingResults with internal metrics, data and path to run folder
        :param epoch_zoom:  Optional list with ranges for zooming loss data
        :return:            None
        """
        self._plotter.loss(
            data=results,
            loss_type=self._settings_model.loss,
            epoch_zoom=epoch_zoom,
            show_plot=False
        )
        self._plotter.custom_loss(
            data=results,
            epoch_zoom=epoch_zoom,
            show_plot=False
        )
        self._plotter.statistics(
            data=results,
            show_plot=False
        )
        match self._settings_ml.mode_train:
            case 0:
                self._plotter.performance_classifier(
                    data=results,
                    show_plot=self._settings_ml.do_block
                )
            case 1:
                if self._settings_data.data_type.lower() == 'mnist':
                    self._plotter.performance_autoencoder_mnist(
                        data=results,
                        show_plot=self._settings_ml.do_block
                    )
                else:
                    self._plotter.performance_autoencoder(
                        data=results,
                        mean_value=self.get_dataset().mean,
                        show_plot=self._settings_ml.do_block
                    )
            case _:
                raise NotImplementedError

    @staticmethod
    def _load_results(path2file: Path) -> TrainingResults:
        if not path2file.is_file():
            raise AttributeError(f"{path2file} is not a file")
        if not path2file.exists():
            raise AttributeError(f"{path2file} does not exists")

        data = np.load(path2file, allow_pickle=True).flatten()[0]
        return data

    def read_file_and_plot(self, path2file: Path, epoch_zoom=None) -> TrainingResults:
        """Loading results file from training and plot the results
        :param path2file:   Path to file with results from PyTorch Training
        :param epoch_zoom:  Optional list or tuple with zoom on specific epoch range
        :return:            None
        """
        data = self._load_results(path2file)
        self.do_plot_results(
            results=data,
            epoch_zoom=epoch_zoom
        )
        return data
