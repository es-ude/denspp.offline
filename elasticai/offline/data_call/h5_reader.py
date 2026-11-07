from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class H5Reader:
    def __init__(self, path2file: Path) -> None:
        """Initialize a reader for an HDF5 file."""
        self._path2file = Path(path2file)

        if not self._path2file.exists():
            raise FileNotFoundError("File not found. Check the file path and current working directory.")

        if not self._path2file.is_file():
            raise FileNotFoundError(f"Path does not refer to a file: {self._path2file}")

        if not h5py.is_hdf5(self._path2file):
            raise ValueError(f"'{self._path2file}' is not a valid HDF5 file.")

    @property
    def path(self) -> Path:
        """Return the path to the HDF5 file."""
        return self._path2file

    def read_all(
        self,
        decode: bool = True,
        chunk_size: int = 1000,
    ) -> dict[str, Any]:
        """Read the HDF5 file structure without loading datasets into memory.

        Each dataset is represented by a callable. Calling it returns a new
        generator that streams the dataset along its first axis.

        Args:
            decode: If True, decode byte strings and NumPy scalar values.
            chunk_size: Number of rows per streamed chunk. Must be positive.

        Returns:
            A nested dictionary representing the HDF5 file structure. Each
            dataset is represented by a callable returning a generator of
            ``(start, end, chunk_data)`` tuples.

        Raises:
            ValueError: If chunk_size is not positive.
        """
        self._validate_chunk_size(chunk_size)

        with h5py.File(self._path2file, "r") as h5f:
            return self._build_lazy_tree(h5f, decode, chunk_size)

    def read_entry(
        self,
        path2entry: Path,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        decode: bool = True,
        stream: bool = True,
        chunk_size: int = 1000,
    ) -> Any:
        """Read a dataset or group from the HDF5 file.

        Datasets can either be streamed in chunks or loaded into memory.
        Groups can only be read in non-streaming mode.

        Args:
            path2entry: Path to the entry inside the HDF5 file, for example
                ``Path("group/subgroup/dataset")``.
            start: First index to read, inclusive. Only used when stream is
                False.
            stop: Last index to read, exclusive. Only used when stream is
                False.
            step: Step size between indices. Only used when stream is False.
            decode: If True, decode byte strings and NumPy scalar values.
            stream: If True, return a generator that streams a dataset in
                chunks. Streaming cannot be used for groups or combined with
                slicing.
            chunk_size: Number of rows per chunk when streaming. Must be
                positive.

        Returns:
            A generator yielding ``(start, end, chunk_data)`` tuples when
            streaming. Otherwise, returns a dataset value, NumPy array, or
            nested dictionary for a group.

        Raises:
            KeyError: If path2entry does not exist in the file.
            TypeError: If streaming is requested for a group.
            ValueError: If slicing is combined with streaming, a scalar
                dataset is sliced, or chunk_size is not positive.
        """
        entry_path = self._to_hdf5_path(path2entry)

        if stream:
            if start is not None or stop is not None or step is not None:
                raise ValueError(
                    "stream=True does not support slicing. Set stream=False "
                    "or use iter_rows() without slicing."
                )

            return self.iter_rows(
                path2entry,
                chunk_size=chunk_size,
                decode=decode,
            )

        with h5py.File(self._path2file, "r") as h5f:
            try:
                entry = h5f[entry_path]
            except KeyError as error:
                raise KeyError(f"'{entry_path}' not found in {self._path2file}") from error

            if isinstance(entry, h5py.Group):
                return self._read_node(entry, decode)

            if start is None and stop is None and step is None:
                return self._read_node(entry, decode)

            if entry.shape == ():
                raise ValueError(f"Dataset '{entry_path}' is scalar and cannot be sliced.")

            data = entry[slice(start, stop, step), ...]
            return self._decode_data(data) if decode else data

    def show_dataset_structure(
        self,
        max_depth: int | None = None,
        show_attrs: bool = False,
    ) -> str:
        """Return an indented tree of the HDF5 file structure.

        Args:
            max_depth: Maximum hierarchy depth to display. If None, display
                the complete structure.
            show_attrs: If True, include dataset attributes.

        Returns:
            A formatted string containing groups, datasets, shapes, data
            types, and optionally dataset attributes.
        """
        with h5py.File(self._path2file, "r") as h5f:
            lines = [self._path2file.name]

            def walk(
                obj: h5py.Group | h5py.Dataset,
                name: str,
                depth: int,
            ) -> None:
                indent = "    " * depth
                prefix = "├── " if depth > 0 else ""

                if max_depth is not None and depth > max_depth:
                    lines.append(f"{indent}└── {name} ... (truncated at depth {max_depth})")
                    return

                if isinstance(obj, h5py.Group):
                    lines.append(f"{indent}{prefix}{name} [group]")

                    for key in sorted(obj.keys()):
                        walk(obj[key], key, depth + 1)

                    return

                info = f"{obj.shape} {obj.dtype}"

                if show_attrs and obj.attrs:
                    attributes = []

                    for key, value in obj.attrs.items():
                        if isinstance(value, bytes):
                            value = value.decode(
                                "utf-8",
                                errors="replace",
                            )

                        attributes.append(f"{key}={value!r}")

                    info += f"  attrs: {{{', '.join(attributes)}}}"

                lines.append(f"{indent}{prefix}{name} [dataset] {info}")

            for key in sorted(h5f.keys()):
                walk(h5f[key], key, 1)

            return "\n".join(lines)

    def iter_rows(
        self,
        path2entry: Path,
        chunk_size: int = 1000,
        decode: bool = True,
    ) -> Iterator[tuple[int, int, Any]]:
        """Stream a dataset along its first axis.

        Args:
            path2entry: Path to the dataset inside the HDF5 file.
            chunk_size: Number of rows per chunk. Must be positive.
            decode: If True, decode byte strings and NumPy scalar values.

        Yields:
            Tuples containing the inclusive start index, exclusive end index,
            and chunk data. A scalar dataset produces one tuple with indices
            zero and one.

        Raises:
            KeyError: If path2entry does not exist in the file.
            TypeError: If path2entry refers to a group.
            ValueError: If chunk_size is not positive.
        """
        entry_path = self._to_hdf5_path(path2entry)
        self._validate_chunk_size(chunk_size)

        with h5py.File(self._path2file, "r") as h5f:
            if entry_path not in h5f:
                raise KeyError(f"'{entry_path}' not found in {self._path2file}")

            entry = h5f[entry_path]

            if not isinstance(entry, h5py.Dataset):
                raise TypeError(f"'{entry_path}' is a group, not a dataset.")

            if entry.shape == ():
                raw_data = entry[()]
                data = self._decode_data(raw_data) if decode else raw_data
                yield 0, 1, data
                return

            length = entry.shape[0]

            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                raw_chunk = entry[start:end]
                chunk = self._decode_data(raw_chunk) if decode else np.asarray(raw_chunk)

                yield start, end, chunk

    @staticmethod
    def _read_node(
        node: h5py.Group | h5py.Dataset,
        decode: bool,
    ) -> Any:
        if isinstance(node, h5py.Dataset):
            data = node[()]
            return H5Reader._decode_data(data) if decode else data

        return {name: H5Reader._read_node(node[name], decode) for name in node.keys()}

    @staticmethod
    def _decode_data(data: Any) -> Any:
        """Decode HDF5 data into Python-friendly types.

        Recursively decode NumPy arrays, byte strings, and NumPy scalar
        values. Scalar arrays are unpacked and other values are returned
        unchanged.
        """
        if isinstance(data, np.ndarray):
            if data.shape == ():
                return H5Reader._decode_data(data[()])

            if data.dtype.kind in {"S", "O"}:
                return np.array(
                    [H5Reader._decode_data(item) for item in data],
                    dtype=object,
                )

            return np.asarray(data)

        if isinstance(data, bytes):
            return data.decode("utf-8")

        if isinstance(data, np.bytes_):
            return data.decode("utf-8")

        if isinstance(data, np.generic):
            return data.item()

        return data

    def _build_lazy_tree(
        self,
        node: h5py.Group | h5py.Dataset,
        decode: bool,
        chunk_size: int,
    ) -> Any:
        """Build a lazy tree in which datasets are callables."""
        if isinstance(node, h5py.Group):
            return {
                key: self._build_lazy_tree(
                    node[key],
                    decode,
                    chunk_size,
                )
                for key in node
            }

        dataset_path = Path(node.name)

        return lambda: self.iter_rows(
            dataset_path,
            chunk_size=chunk_size,
            decode=decode,
        )

    @staticmethod
    def _to_hdf5_path(path2entry: Path) -> str:
        """Convert a Path to an HDF5-compatible internal path."""
        return path2entry.as_posix()

    @staticmethod
    def _validate_chunk_size(chunk_size: int) -> None:
        """Validate the number of rows per chunk."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
