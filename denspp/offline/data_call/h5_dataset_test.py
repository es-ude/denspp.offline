from pathlib import Path
from random import randint
from shutil import rmtree

import h5py
import numpy as np
import pytest

from denspp.offline import get_path_to_project

from .h5_dataset import CollectorH5, LabelCollector

num_samples = 2**10


@pytest.fixture(scope="session", autouse=False)
def path():
    path2testfolder = Path(get_path_to_project()) / "temp_test" / "h5_dataset"
    if path2testfolder.exists():
        rmtree(path2testfolder)
    path2testfolder.mkdir(parents=True)
    yield path2testfolder


def test_label_collecting_string():
    dut = LabelCollector()
    assert dut._is_empty

    ret = dut.add("A")
    assert not dut._is_empty
    assert ret == 0

    ret = dut.add("A")
    assert not dut._is_empty
    assert ret == 0

    ret = dut.add("B")
    assert not dut._is_empty
    assert ret == 1

    ret = dut.add("A")
    assert not dut._is_empty
    assert ret == 0

    assert dut.get_all() == {"A": 0, "B": 1}
    assert dut.get_keys() == ["A", "B"]
    assert dut.get_values() == [0, 1]


def test_label_collecting_integer():
    dut = LabelCollector()
    assert dut._is_empty

    ret = dut.add(0)
    assert not dut._is_empty
    assert ret == 0

    ret = dut.add(0)
    assert not dut._is_empty
    assert ret == 0

    ret = dut.add(1)
    assert not dut._is_empty
    assert ret == 1

    ret = dut.add(0)
    assert not dut._is_empty
    assert ret == 0

    assert dut.get_all() == {"0": 0, "1": 1}
    assert dut.get_keys() == ["0", "1"]
    assert dut.get_values() == [0, 1]


@pytest.mark.parametrize("n_chunks", [1, 8, 16, 512])
def test_h5_data_collection_integer(path: Path, n_chunks: int):
    path2file = path / f"test_data_int_{n_chunks:02d}.h5"

    data_in = [randint(a=-100, b=100) for _ in range(num_samples)]

    with h5py.File(path2file, "w") as h5f:
        dut = CollectorH5(h5f, name="data", chunks=n_chunks)
        dut.define_datatype(int)

        for val in data_in:
            dut.add(val)
        h5f.close()

    with h5py.File(path2file, "r") as h5f:
        keys = h5f.keys()
        assert list(keys) == ["data"]

        data: np.ndarray = h5f["data"][()]
        assert data.tolist() == data_in
        h5f.close()


@pytest.mark.parametrize("n_chunks", [1, 8, 16, 512])
def test_h5_data_collection_string(path: Path, n_chunks: int):
    path2file = path / f"test_data_str_{n_chunks:02d}.h5"

    data_in = ["AFDSAD" for _ in range(num_samples)]

    with h5py.File(path2file, "w") as h5f:
        dut = CollectorH5(h5f, name="data", chunks=n_chunks)
        dut.define_datatype(str)

        for val in data_in:
            dut.add(val)
        h5f.close()

    with h5py.File(path2file, "r") as h5f:
        keys = h5f.keys()
        assert list(keys) == ["data"]

        data: np.ndarray = np.array(h5f["data"][()], dtype=str)
        assert data.tolist() == data_in
        h5f.close()


@pytest.mark.parametrize("n_chunks", [1, 8, 16, 512])
def test_h5_data_collection_list(path: Path, n_chunks: int):
    path2file = path / f"test_data_list_{n_chunks:02d}.h5"

    data_in = [[randint(a=-10, b=10) for _ in range(10)] for _ in range(num_samples)]

    with h5py.File(path2file, "w") as h5f:
        dut = CollectorH5(h5f, name="data", chunks=n_chunks)
        dut.define_datatype(list)

        for val in data_in:
            dut.add(val)
        h5f.close()

    with h5py.File(path2file, "r") as h5f:
        keys = h5f.keys()
        assert list(keys) == ["data"]

        data: np.ndarray = np.array(h5f["data"][()], dtype=int)
        assert data.tolist() == data_in
        h5f.close()


@pytest.mark.parametrize("n_chunks", [1, 8, 16, 512])
def test_h5_data_collection_numpy(path: Path, n_chunks: int):
    path2file = path / f"test_data_numpy_{n_chunks:02d}.h5"

    data_in = [[randint(a=-128, b=127) for _ in range(10)] for _ in range(num_samples)]

    with h5py.File(path2file, "w") as h5f:
        dut = CollectorH5(h5f, name="data", chunks=n_chunks)
        dut.define_datatype(np.int8)

        for val in data_in:
            dut.add(val)
        h5f.close()

    with h5py.File(path2file, "r") as h5f:
        keys = h5f.keys()
        assert list(keys) == ["data"]

        data: np.ndarray = np.array(h5f["data"][()])
        assert data.tolist() == data_in
        h5f.close()


if __name__ == "__main__":
    pytest.main()
