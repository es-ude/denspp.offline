from typing import Any

import h5py


class LabelCollector:
    _labels: dict

    def __init__(self) -> None:
        """Class for collecting labels of a dataset and build a dictionary with names"""
        self._labels = dict()

    @property
    def _is_empty(self) -> bool:
        return len(self._labels) == 0

    def add(self, new_data: int | str) -> int:
        """Adds a new label to the dictionary
        :param new_data:    new label
        :return:            id of the new label
        """
        val_input = f"{new_data}"
        keys = self._labels.keys()

        if self._is_empty:
            self._labels = {val_input: len(keys)}
        elif val_input not in keys:
            self._labels.update({val_input: len(keys)})
        return self._labels[val_input]

    def get_all(self) -> dict[str, Any]:
        """Returns the dictionary with all labels"""
        if self._is_empty:
            raise ValueError("No labels in the dictionary")
        return self._labels

    def get_keys(self) -> list[str]:
        """Returns the keys of the dictionary"""
        if self._is_empty:
            raise ValueError("No labels in the dictionary")
        return list(self._labels.keys())

    def get_values(self) -> list[Any]:
        """Returns the values of the dictionary"""
        if self._is_empty:
            raise ValueError("No labels in the dictionary")
        return list(self._labels.values())


class CollectorH5:
    _data: h5py.Dataset
    _file: h5py.File
    _name: str
    _chunks: int
    _datatype: None

    def __init__(self, h5_linker: h5py.File, name: str, chunks: int = 8) -> None:
        """Registering a collector buffer to write into h5 file
        :param h5_linker:   h5py.File object
        :param name:        name of the dataset
        :param chunks:      chunk size
        :return:            None
        """
        self._file = h5_linker
        self._name = name
        self._chunks = chunks

    @property
    def get_number_samples(self) -> int:
        """Returns the number of samples in the buffer"""
        return self._data.shape[0]

    def define_datatype(self, datatype: Any) -> None:
        """Defines the datatype of the buffer
        :param datatype:    type of used datatype in the buffer
        """
        if datatype == str:
            self._datatype = h5py.string_dtype(encoding="utf-8")
        elif datatype == list:
            self._datatype = None
        else:
            self._datatype = datatype

    def add(self, data: int | str | float | list) -> None:
        """Adds a new sample to the buffer
        :param data:        new sample
        """
        if not hasattr(self, "_data"):
            if isinstance(data, (int, str, float)):
                shape = (0,)
                chunkshape = (self._chunks,)
                maxshape = (None,)
            else:
                shape = (0, len(data))
                chunkshape = (self._chunks, len(data))
                maxshape = (None, len(data))

            self._data = self._file.create_dataset(
                name=self._name,
                chunks=chunkshape,
                shape=shape,
                maxshape=maxshape,
                dtype=self._datatype,
                compression="gzip",
                compression_opts=4,
            )
        old_len = self.get_number_samples
        self._data.resize(old_len + 1, axis=0)
        self._data[old_len:] = data
