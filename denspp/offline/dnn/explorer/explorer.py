import json
import logging
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from logging import Logger, getLogger
from pathlib import Path
from typing import Any

import numpy as np
from elasticai.explorer.hw_nas import hw_nas, reconstruct_model_from_json
from elasticai.explorer.hw_nas.estimators import ParamEstimator, TrainMetricsEstimator
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from elasticai.explorer.training.data import DatasetSpecification
from elasticai.explorer.training.trainer import SupervisedTrainer, accuracy_fn
from torch import Tensor, argmax, from_numpy, nn
from torch.utils.data import Dataset

from denspp.offline import get_path_to_project
from denspp.offline.data_format import JsonHandler, YamlHandler
from denspp.offline.dnn import (
    DatasetFromFile,
    DefaultSettingsDataset,
    DefaultSettingsTraining,
    PyTorchTrainer,
    SettingsDataset,
)
from denspp.offline.dnn.model_library import DatasetLoaderLibrary
from denspp.offline.dnn.training import DefaultSettingsTrainingCE, SettingsClassifier, TrainClassifier

from .settings import (
    DefaultSettingsExplorer,
    DefaultSettingsSearchSpace,
    SettingsExplorer,
    SettingsSearchSpace,
)


class ExplorerModel(nn.Module):
    def __init__(self, model: nn.Module, shape: tuple):
        super().__init__()
        self.model_shape = shape
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        prob = self.model(x)
        return prob, argmax(prob, dim=1)


class DatasetClassifierExplorer(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()
        self.data = from_numpy(data)
        self.labels = from_numpy(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.data[index], self.labels[index]


class ExploreClassifier:
    _sets_exp: SettingsExplorer
    _sets_data: SettingsDataset
    _search_space: SettingsSearchSpace
    _logger: Logger
    _logger: Logger
    _device: str
    _path2config: Path

    def __init__(
        self, settings: SettingsExplorer = DefaultSettingsExplorer, path2config: Path = Path("./config")
    ) -> None:
        """Routine for exploring a HW-aware model and its full training
        :param settings:    Settings object for defining the search space
        :param path2config: Path to the config file
        :return:            None
        """
        logging.getLogger("optuna").setLevel(logging.WARNING)
        logging.getLogger("explorer").setLevel(logging.WARNING)
        self._logger = getLogger(self.__class__.__name__)
        self._path2config = path2config

        self._sets_exp = JsonHandler(
            template=settings,
            path=path2config,
            file_name=f"ConfigExplorer_{settings.experiment_name.capitalize()}",
        ).get_class(SettingsExplorer)

        self._device = TrainClassifier(
            config_train=DefaultSettingsTrainingCE,
            config_data=DefaultSettingsDataset,
            do_train=False,
            device_num=self._sets_exp.device,
        ).get_device

    @staticmethod
    def _select_indices(labels: np.ndarray, limit: int, seed: int, shuffle: bool) -> np.ndarray:
        reduce_data = 0 < limit < labels.shape[0]
        use_samples = limit if reduce_data else labels.shape[0]

        if not reduce_data and not shuffle:
            return np.arange(labels.shape[0])
        elif reduce_data and not shuffle:
            return np.arange(limit)
        else:
            rng = np.random.default_rng(seed)
            unique_labels = np.unique(labels)
            per_class = max(1, use_samples // unique_labels.size)
            selected: list[np.ndarray] = []

            for label in unique_labels:
                label_indices = np.flatnonzero(labels == label)
                take = min(per_class, label_indices.size)
                selected.append(rng.choice(label_indices, size=take, replace=False))

            indices = np.concatenate(selected)
            if indices.size < use_samples:
                remaining = np.setdiff1d(np.arange(labels.shape[0]), indices, assume_unique=False)
                extra = rng.choice(
                    remaining,
                    size=min(use_samples - indices.size, remaining.size),
                    replace=False,
                )
                indices = np.concatenate([indices, extra])

            rng.shuffle(indices)
            return indices

    def _init_folders(self) -> Path:
        date = datetime.now().strftime("%Y%m%d_%H%M%S_explore_")
        folder_name = date + self._sets_exp.experiment_name
        experiment_dir = (get_path_to_project("runs") / folder_name).resolve().absolute()
        experiment_dir.mkdir(exist_ok=True, parents=True)
        return experiment_dir

    def load_search_space(self) -> None:
        """Loading the pre-defined search space definition from a YAML file"""
        self._search_space = YamlHandler(
            template=DefaultSettingsSearchSpace, path=self._path2config, file_name="search_space"
        ).get_class(SettingsSearchSpace)

    def load_search_space_from_file(self, path2yaml: Path) -> None:
        """Loading the search space definition from a YAML file
        :param path2yaml:   Path to the YAML file with the search space definition
        :return:            None
        """
        if path2yaml.suffix not in [".yaml", ".yml"]:
            raise AttributeError("YAML-file with search space definition must end with .yaml or .yml")
        self._search_space = YamlHandler(
            template=DefaultSettingsSearchSpace, path=path2yaml.parent, file_name=path2yaml.name
        ).get_class(SettingsSearchSpace)

    def load_search_space_direct(self, search_space: SettingsSearchSpace) -> None:
        """Loading the search space definition from a YAML file
        :param search_space:    Search space definition with dataclass SettingsSearchSpace
        :return:                None
        """
        self._search_space = search_space
        YamlHandler(
            template=search_space, path=self._path2config, file_name="search_space.yaml"
        ).write_to_yaml()

    @staticmethod
    def _get_dataset_loader() -> Any:
        datalib = DatasetLoaderLibrary().get_registry()
        matches = [item for item in datalib.get_library_overview() if "DatasetLoader" == item]
        if len(matches) == 0:
            raise AttributeError("No DatasetLoader available")
        return datalib.build_object(matches[0])

    def _load_data(self) -> DatasetFromFile:
        sets_default: SettingsDataset = deepcopy(DefaultSettingsDataset)
        sets_default.data_type = self._sets_exp.experiment_name.lower()

        self._sets_data = JsonHandler(
            template=sets_default,
            path=self._path2config,
            file_name="ConfigDataset_Explorer",
        ).get_class(SettingsDataset)
        return self._get_dataset_loader()(self._sets_data).load_dataset()

    def prepare_data(self, do_shuffle: bool = True, add_dimension: bool = False) -> DatasetSpecification:
        """Function for preparing the dataset for training using the elasticAI.explorer
        :param do_shuffle:      Boolean for shuffling the dataset samples
        :param add_dimension:   Boolean to add a dimension to the dataset in order to train Conv1d from transient data
        :return:                Tuned dataset for training
        """
        dataset = self._load_data()
        indices = self._select_indices(
            labels=dataset.label,
            limit=self._sets_exp.num_data_samples,
            seed=self._sets_exp.seed,
            shuffle=do_shuffle,
        )
        data_new = dataset.data[indices].astype(np.float32)
        if add_dimension:
            data_new = np.expand_dims(data_new, axis=1)
        label_new = dataset.label[indices].astype(np.int64)

        data = DatasetClassifierExplorer(data=data_new, labels=label_new)
        return DatasetSpecification(dataset=data, shuffle=do_shuffle, split_seed=self._sets_exp.seed)

    def run_search(self, dataset: DatasetSpecification, loss_fn=nn.CrossEntropyLoss()) -> Path:
        """Executing the search space for finding the best performing model
        :param dataset:         Dataset used for the search
        :param loss_fn:         Loss function to use for training
        :return:                Path to the folder containing results
        """
        path2temp = self._init_folders()
        self._logger.info("Using computing device: %s", self._device)

        # --- Prepare Search Space Engine
        trainer = SupervisedTrainer(
            device=self._device,
            dataset_spec=dataset,
            loss_fn=loss_fn,
            batch_size=self._sets_exp.batch_size,
            extra_metrics={"accuracy": accuracy_fn},
        )

        criteria = OptimizationCriteria()
        criteria.register_objective(
            estimator=TrainMetricsEstimator(
                trainer=trainer,
                metric_name="accuracy",
                n_estimation_epochs=self._sets_exp.num_epochs_trial,
                learning_rate=self._sets_exp.learning_rate,
            ),
            transform=None,
            weight=1.0,
        )
        criteria.register_objective(
            estimator=ParamEstimator(),
            transform=None,
            weight=0.0,
        )
        # criteria.register_objective(
        #    estimator=FLOPsEstimator(),
        #    transform=None,
        #    weight=0.0,
        # )

        if not hasattr(self, "_search_space"):
            raise RuntimeError("Explorer did not find a valid search space.")
        top_models, model_configs, model_metrics = hw_nas.search(
            search_space_cfg=asdict(self._search_space),
            search_strategy=self._sets_exp.get_search_strategy,
            optimization_criteria=criteria,
            hw_nas_parameters=HWNASParameters(
                max_search_trials=self._sets_exp.num_trials_search,
                top_n_models=self._sets_exp.num_best_models,
                count_only_completed_trials=False,
            ),
        )
        if not top_models:
            raise RuntimeError("Explorer did not find a valid model.")
        self._logger.info(top_models)

        # --- Logging metrics of best models
        for model, metrics in zip(top_models, model_metrics):
            metrics.update(
                {
                    "dataset_size": len(dataset.dataset),
                    "raw_dataset_size": dataset.dataset[0][0].shape,
                }
            )
        with open(path2temp / "models.json", "w") as f:
            json.dump(model_configs, f, indent=2)
        with open(path2temp / "metrics.json", "w") as f:
            json.dump(model_metrics, f, indent=2)

        YamlHandler(
            template=self._search_space, path=path2temp, file_name="search_space.yaml"
        ).write_to_yaml()
        return path2temp

    def run_full_training(
        self, path2run: Path, shuffle_data: bool = True, add_dimension: bool = True
    ) -> None:
        """Executing the full training of the best model after search space execution
        :param path2run:        Path to the folder containing the search space results
        :param shuffle_data:    Boolean for shuffling the dataset samples
        :param add_dimension:   Boolean to add a dimension to the dataset in order to train Conv1d from transient data
        :return:                None
        """
        path2save = (path2run / "full").resolve()
        sets_train: SettingsClassifier = deepcopy(DefaultSettingsTrainingCE)
        sets_train.data_type = self._sets_exp.experiment_name.lower()
        sets_train.batch_size = self._sets_exp.batch_size
        sets_train.learning_rate = self._sets_exp.learning_rate
        sets_train.num_epochs = self._sets_exp.num_epochs_best
        sets_train.model_name = ""
        sets_train.data_do_shuffle = shuffle_data
        sets_train.deterministic_do = True
        sets_train.deterministic_seed = self._sets_exp.seed
        sets_train.patience = 100

        trainer = TrainClassifier(
            config_train=sets_train,
            config_data=DefaultSettingsDataset,
            do_train=True,
            device_num=self._sets_exp.device,
        )
        dataset = self._load_data()
        if add_dimension:
            dataset = DatasetFromFile(
                data=np.expand_dims(dataset.data, axis=1),
                label=dataset.label,
                dict=dataset.dict,
                mean=dataset.mean,
            )

        trainer.load_dataset(dataset=dataset)
        data_shape = (1,) + dataset.data.shape[1:]

        best_model = reconstruct_model_from_json(
            file2json=(path2run / "models.json").resolve(),
            file2search=(path2run / "search_space.yaml").resolve(),
        )
        trainer.load_model(model=ExplorerModel(model=best_model, shape=data_shape))

        metrics = trainer.do_training(
            path2save=path2save,
        )
        results = trainer.do_post_training_validation(do_ptq=False)

        plotter = PyTorchTrainer(
            use_case="Explorer",
            settings=DefaultSettingsTraining,
            default_model="",
        )
        plotter._settings_model = sets_train
        results = plotter._save_training_results(
            addon="",
            metrics=metrics,
            data_result=results,
            custom_metrics=[],
            path2save=path2save,
        )
        plotter.do_plot_results(results=results, do_plot=True)
