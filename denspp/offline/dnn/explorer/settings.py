from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from elasticai.explorer.hw_nas.hw_nas import SearchStrategy

from denspp.offline.dnn.training.common_train import TrainingsDevice


class ExploreStrategySearch(IntEnum):
    random = 0
    evolution = 1


@dataclass
class SettingsSearchSpace:
    """Search Space Definition for exploring the HW-aware Neural Network
    Attributes:
        input:              Input shape of the data for the model
        output:             Output shape of the model
        sequence:           List with description of the blocks / sequences in the model
        default_op_params:  Dictionary with hyperparameters of the layers inside the sequences
        composites:         Dictionary with describing the sequences setup
    """

    input: int | list[int]
    output: int
    sequence: list[dict]
    default_op_params: dict[str, Any]
    composites: dict[str, dict[str, Any]]


DefaultSettingsSearchSpace = SettingsSearchSpace(
    input=[1, 28, 28],
    output=10,
    sequence=[
        {
            "block": "1",
            "op_candidates": "classifier",
            "type_repeat": {"type": "repeat_params", "depth": [1, 2]},
        },
        {"block": "2", "op_candidates": "linear"},
    ],
    default_op_params={
        "linear": {"width": [64, 96, 128, 160, 192]},
        "batch_norm": None,
        "activation": {"op": ["relu", "prelu"]},
    },
    composites={
        "classifier": {
            "sequence": [
                {"block": "classifier_1", "op_candidates": "linear"},
                {"block": "classifier_2", "op_candidates": "batch_norm"},
                {"block": "classifier_3", "op_candidates": "activation"},
            ]
        }
    },
)


@dataclass
class SettingsExplorer:
    """Settings for defining the model exploration using the elasticAI.explorer engine
    Attributes:
        experiment_name:    String with experiment name for saving
        num_data_samples:   Number of data samples to use (0=taking all)
        num_trials_search:  Number of model exploration trials
        num_epochs_trial:   Number of training epochs for each model candidate during search
        num_epochs_best:    Number of training epochs in the final stage (taking best model)
        batch_size:         Number of samples per batch
        learning_rate:      Learning rate for optimizer
        seed:               Seed for random number generator
        num_best_models:    Number of best models to return
        device:             Training device
        search_strategy:    String with indicating the Optuna Optimizer
    """

    experiment_name: str
    num_data_samples: int
    num_trials_search: int
    num_epochs_trial: int
    num_epochs_best: int
    batch_size: int
    learning_rate: float
    seed: int
    num_best_models: int
    device: TrainingsDevice | int
    search_strategy: ExploreStrategySearch | int

    @property
    def get_search_strategy(self) -> SearchStrategy:
        """Returning the right SearchStrategy definition for the elasticAI.explorer"""
        match self.search_strategy:
            case ExploreStrategySearch.random:
                return SearchStrategy.RANDOM_SEARCH
            case ExploreStrategySearch.evolution:
                return SearchStrategy.EVOLUTIONARY_SEARCH
            case _:
                raise ValueError("Selected strategy is unknown")


DefaultSettingsExplorer = SettingsExplorer(
    experiment_name="mnist",
    num_data_samples=0,
    num_trials_search=30,
    num_epochs_trial=10,
    num_epochs_best=100,
    batch_size=512,
    learning_rate=0.1,
    seed=42,
    num_best_models=1,
    device=TrainingsDevice.auto,
    search_strategy=ExploreStrategySearch.random,
)
