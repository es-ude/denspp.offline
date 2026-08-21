import shutil
from copy import deepcopy
from os import remove
from pathlib import Path

import pytest
import torch

from denspp.offline import get_path_to_project

from .explorer import (
    DefaultSettingsExplorer,
    DefaultSettingsSearchSpace,
    ExploreClassifier,
    SettingsExplorer,
)


@pytest.fixture
def sets() -> SettingsExplorer:
    sets: SettingsExplorer = deepcopy(DefaultSettingsExplorer)
    sets.experiment_name = "waveforms"
    return sets


@pytest.fixture
def path2temp() -> Path:
    path = get_path_to_project("temp_test")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def test_init(sets: SettingsExplorer, path2temp: Path) -> None:
    ExploreClassifier(settings=sets, path2config=path2temp)
    assert (path2temp / "ConfigExplorer_Waveforms.json").exists()


def test_load_search_space(sets: SettingsExplorer, path2temp: Path) -> None:
    dut = ExploreClassifier(settings=sets, path2config=path2temp)

    path2search = path2temp / "search_space.yaml"
    if path2search.exists():
        remove(path2search, ignore_errors=True)

    dut.load_search_space_direct(search_space=DefaultSettingsSearchSpace)
    assert dut._search_space == DefaultSettingsSearchSpace
    assert path2search.exists()

    dut._search_space = None
    dut.load_search_space_from_file(path2yaml=path2search)
    assert dut._search_space == DefaultSettingsSearchSpace

    dut._search_space = None
    dut.load_search_space()
    assert dut._search_space == DefaultSettingsSearchSpace


@pytest.mark.slow
@pytest.mark.parametrize("datatype", ["mnist", "waveforms"])
def test_load_data(sets: SettingsExplorer, path2temp: Path, datatype: str) -> None:
    sets.experiment_name = datatype
    data = ExploreClassifier(settings=sets, path2config=path2temp)._load_data()

    if datatype == "mnist":
        assert data.data.shape == (70000, 28, 28)
        assert data.label.shape == (70000,)
        assert len(data.dict) == 10
    else:
        assert data.data.shape == (12000, 280)
        assert data.label.shape == (12000,)
        assert len(data.dict) == 12


@pytest.mark.slow
def test_prepare_data_full(sets: SettingsExplorer, path2temp: Path) -> None:
    data = ExploreClassifier(settings=sets, path2config=path2temp).prepare_data(do_shuffle=False)
    assert len(data.dataset) == 12000
    assert list(data.dataset[0][0].shape) == [
        280,
    ]
    assert not data.shuffle
    assert data.train_val_test_ratio == [0.7, 0.1, 0.2]
    assert data.split_seed == sets.seed


@pytest.mark.slow
def test_prepare_data_shuffling(sets: SettingsExplorer, path2temp: Path) -> None:
    dut = ExploreClassifier(settings=sets, path2config=path2temp)

    data_unshuffled = dut.prepare_data(do_shuffle=False)
    data_shuffled = dut.prepare_data(do_shuffle=True)

    assert len(data_unshuffled.dataset) == len(data_shuffled.dataset)
    assert not torch.equal(data_unshuffled.dataset[0][0], data_shuffled.dataset[0][0])


@pytest.mark.slow
def test_prepare_data_reducing(sets: SettingsExplorer, path2temp: Path) -> None:
    sets.num_data_samples = 100
    data = ExploreClassifier(settings=sets, path2config=path2temp).prepare_data(do_shuffle=False)

    assert len(data.dataset) == sets.num_data_samples
    assert list(data.dataset[0][0].shape) == [
        280,
    ]
    assert not data.shuffle
    assert data.train_val_test_ratio == [0.7, 0.1, 0.2]
    assert data.split_seed == sets.seed


@pytest.mark.slow
def test_run_search_without_search_space(sets: SettingsExplorer, path2temp: Path) -> None:
    sets.num_data_samples = 1024
    sets.search_strategy = 0
    sets.num_trials_search = 2
    sets.num_epochs_trial = 5

    dut = ExploreClassifier(settings=sets, path2config=path2temp)
    data = dut.prepare_data(do_shuffle=True)
    try:
        dut.run_search(
            dataset=data,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )
    except RuntimeError:
        assert True
    else:
        assert False


@pytest.mark.slow
def test_run_search_with_search_space(sets: SettingsExplorer, path2temp: Path) -> None:
    sets.search_strategy = 0
    sets.num_trials_search = 3
    sets.num_epochs_trial = 5

    dut = ExploreClassifier(settings=sets, path2config=path2temp)
    data = dut.prepare_data(do_shuffle=True)

    dut.load_search_space()
    dut._search_space.input = data.dataset[0][0].size(0)
    dut._search_space.output = 12

    dut.run_search(
        dataset=data,
        loss_fn=torch.nn.CrossEntropyLoss(),
    )


@pytest.mark.slow
def test_run_full(sets: SettingsExplorer, path2temp: Path) -> None:
    sets.search_strategy = 0
    sets.num_trials_search = 2
    sets.num_epochs_trial = 5
    sets.num_epochs_best = 10

    dut = ExploreClassifier(settings=sets, path2config=path2temp)
    data = dut.prepare_data(do_shuffle=True)

    dut.load_search_space()
    dut._search_space.input = data.dataset[0][0].size(0)
    dut._search_space.output = 12

    path2temp = dut.run_search(
        dataset=data,
        loss_fn=torch.nn.CrossEntropyLoss(),
    )
    dut.run_full_training(path2run=path2temp, shuffle_data=True)
