from typing import Any

import h5py
import numpy as np


class LabelCollector:
    _labels: dict[str, int]

    def __init__(self) -> None:
        """Class for collecting labels of a dataset and building a dictionary with names."""
        self._labels = {}

    @property
    def _is_empty(self) -> bool:
        return not self._labels

    def add(self, new_data: int | str) -> int:
        """Adds a new label to the dictionary.

        :param new_data:    new label (int or str)
        :return:            id of the new label (integer)
        """
        val_input = str(new_data)
        if val_input not in self._labels:
            self._labels[val_input] = len(self._labels)
        return self._labels[val_input]

    def get_all(self) -> dict[str, int]:
        """Returns a copy of the dictionary with all labels."""
        if self._is_empty:
            raise ValueError("No labels in the dictionary")
        return self._labels.copy()

    def get_keys(self) -> list[str]:
        """Returns the keys (label names) of the dictionary."""
        if self._is_empty:
            raise ValueError("No labels in the dictionary")
        return list(self._labels.keys())

    def get_values(self) -> list[int]:
        """Returns the values (label IDs) of the dictionary."""
        if self._is_empty:
            raise ValueError("No labels in the dictionary")
        return list(self._labels.values())


class CollectorH5:
    _data: h5py.Dataset | None
    _file: h5py.File
    _name: str
    _chunks: int
    _datatype: Any | None
    _expected_shape: tuple[int, ...] | None

    def __init__(self, h5_linker: h5py.File, name: str, chunks: int = 8) -> None:
        """Register a collector buffer to write into an HDF5 file.

        If a dataset with the given name already exists, it will be reused
        provided it is chunked and resizable along the first axis. Otherwise,
        a new dataset is created on the first call to `add()`.

        :param h5_linker:   open h5py.File object
        :param name:        name of the dataset
        :param chunks:      chunk size for the first axis
        :return:            None
        """
        self._file = h5_linker
        self._name = name
        self._chunks = chunks
        self._data = None
        self._datatype = None
        self._expected_shape = None

        if name in h5_linker:
            existing = h5_linker[name]
            if not isinstance(existing, h5py.Dataset):
                raise TypeError(f"'{name}' exists but is not a Dataset")
            if existing.chunks is None:
                raise ValueError(f"Existing dataset '{name}' is not chunked")
            if existing.maxshape is None or existing.maxshape[0] is not None:
                raise ValueError(f"Existing dataset '{name}' is not resizable along axis 0")
            self._data = existing
            self._datatype = existing.dtype
            self._expected_shape = existing.shape[1:]

    @property
    def sample_count(self) -> int:
        """Returns the number of samples currently stored in the buffer."""
        if self._data is None:
            return 0
        return self._data.shape[0]

    def define_datatype(self, datatype: Any) -> None:
        """Defines the datatype of the buffer.

        Must be called before the first call to `add()`. For string data,
        both `str` and `np.str_` are supported and will be stored as
        variable-length UTF-8 strings. For other types, the argument is
        passed to `np.dtype()`.

        :param datatype:    Python type, NumPy dtype, or string representation
        """
        if not self._file.id.valid:
            raise ValueError("The HDF5 file is closed")
        if datatype is str or datatype is np.str_:
            self._datatype = h5py.string_dtype(encoding="utf-8")
        else:
            self._datatype = np.dtype(datatype)

    def add(self, data: Any) -> None:
        """Adds a new sample to the buffer.

        The shape of the first sample determines the expected shape for all
        subsequent samples. If a sample with a different shape is added, a
        ValueError is raised.

        :param data:        new sample (scalar, list, tuple, or numpy array)
        """
        if not self._file.id.valid:
            raise ValueError("The HDF5 file is closed")
        if self._datatype is None:
            raise ValueError(
                "Datatype must be defined before adding data. Please call `define_datatype()` first."
            )

        sample_shape = np.shape(data)

        if self._data is None:
            self._expected_shape = sample_shape
            if len(sample_shape) == 0:
                chunkshape = (self._chunks,)
                maxshape = (None,)
            else:
                chunkshape = (self._chunks,) + sample_shape
                maxshape = (None,) + sample_shape

            self._data = self._file.create_dataset(
                name=self._name,
                chunks=chunkshape,
                shape=(0,) + sample_shape,
                maxshape=maxshape,
                dtype=self._datatype,
                compression="gzip",
                compression_opts=4,
            )
        else:
            if sample_shape != self._expected_shape:
                raise ValueError(f"Shape mismatch: Got {sample_shape}, expected {self._expected_shape}")

        old_len = self.sample_count
        self._data.resize(old_len + 1, axis=0)
        self._data[old_len] = data
