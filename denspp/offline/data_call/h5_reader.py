from pathlib import Path
from typing import Any

import h5py
import numpy as np


class H5Reader:
    def __init__(self, path2file: str | Path) -> None:
        """Reads contents from an HDF5 file."""
        self._path2file = Path(path2file)

        if not self._path2file.exists():
            raise FileNotFoundError(
                "File not found - Please check file path and current working directory."
            )

    @property
    def path(self) -> Path:
        return self._path2file

    def read_all(self, decode: bool = True, chunk_size: int = 1000) -> dict[str, Any]:
        """Read the full HDF5 file structure without loading data into memory.

        Instead of loading datasets, they are replaced by a callable (lambda) that,
        when invoked, returns a generator from iter_rows() for that dataset.
        This makes the method safe for arbitrarily large files.

        Args:
            decode: If True, decode byte strings when accessing data.
            chunk_size: Number of rows per chunk when iterating over a dataset.

        Returns:
            Nested dict where every dataset is represented as a callable
            returning a generator of (start, end, chunk_data).
        """
        with h5py.File(self._path2file, "r") as h5f:
            return self._build_lazy_tree(h5f, decode, chunk_size)

    def read_entry(
        self,
        path2entry: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        decode: bool = True,
        stream: bool = True,
        chunk_size: int = 1000,
    ):
        """Read a specific entry (dataset or group) from the HDF5 file.

        For datasets, you can either stream the data in chunks (default) or
        load a concrete slice into memory.

        Args:
            path2entry: Full path to the entry inside the HDF5 file
                (e.g., "group/subgroup/dataset").
            start: First index to read (inclusive). Only used when stream=False.
            stop: Last index to read (exclusive). Only used when stream=False.
            step: Step size between indices. Only used when stream=False.
            decode: If True, decode byte strings to Python strings.
            stream: If True (default), return a generator from iter_rows().
                    Slicing (start/stop/step) is not allowed in stream mode.
            chunk_size: Rows per chunk, only used when stream=True.

        Returns:
            - If stream=True: a generator yielding (start, end, chunk_data).
            - If stream=False: a decoded Python object (scalar, list, or array)
              for the dataset, or a dictionary for a group.

        Raises:
            KeyError: If `path2entry` does not exist in the file.
            ValueError: If slicing is attempted on a scalar dataset, or if
                        stream=True is used with slicing parameters.
        """
        # Streaming mode: stream dataset rows in chunks – memory efficient for large files.
        if stream:
            if start is not None or stop is not None or step is not None:
                raise ValueError("stream=True does not support slicing, instead use iter_rows directly.")

            with h5py.File(self._path2file, "r") as h5f:
                if path2entry not in h5f:
                    raise KeyError(f"'{path2entry}' not found in {self._path2file}")

            return self.iter_rows(path2entry, chunk_size=chunk_size, decode=decode)

        # Non‑streaming mode: load the whole entry/or slice into memory
        with h5py.File(self._path2file, "r") as h5f:
            try:
                entry = h5f[path2entry]
            except KeyError as e:
                raise KeyError(f"'{path2entry}' not found in {self._path2file}") from e

            if isinstance(entry, h5py.Group):
                return self._read_node(entry)

            if start is None and stop is None and step is None:
                return self._read_node(entry)

            if entry.shape == ():
                raise ValueError(f"Dataset '{path2entry}' is scalar and cannot be sliced.")

            slc = slice(start, stop, step)
            data = entry[slc, ...]
            return self._decode_data(data)

    def show_dataset_structure(self, max_depth: int | None = None, show_attrs: bool = False) -> str:
        """Return an indented tree with dataset shapes/dtypes and optional attributes."""
        with h5py.File(self._path2file, "r") as h5f:
            lines = [self._path2file.name]

            def walk(obj, name, depth):
                indent = "    " * depth
                prefix = "├── " if depth > 0 else ""

                if max_depth is not None and depth > max_depth:
                    lines.append(f"{indent}└── ... (truncated at depth {max_depth})")
                    return

                if isinstance(obj, h5py.Group):
                    lines.append(f"{indent}{prefix}{name} [group]")
                    for key in sorted(obj.keys()):
                        walk(obj[key], key, depth + 1)

                else:
                    info = f"{obj.shape} {obj.dtype}"
                    if show_attrs and obj.attrs:
                        attrs = []
                        for k, v in obj.attrs.items():
                            if isinstance(v, bytes):
                                v = v.decode("utf-8", errors="replace")
                            attrs.append(f"{k}={v!r}")
                        info += f"  attrs: {{{', '.join(attrs)}}}"
                    lines.append(f"{indent}{prefix}{name} [dataset] {info}")

            for key in sorted(h5f.keys()):
                walk(h5f[key], key, 1)

            return "\n".join(lines)

    def iter_rows(self, path2entry: str, chunk_size: int = 1000, decode: bool = True):
        """Stream dataset rows in chunks – memory efficient for large files.

        Args:
            path2entry: Path to the dataset (must be a dataset, not a group).
            chunk_size: Number of rows per chunk.
            decode: If True, decode bytes to strings (default).
                    If False, return raw NumPy arrays without decoding.

        Yields:
            A tuple (start_index, end_index, chunk_data) for each chunk.
            If decode=True, chunk_data contains decoded Python objects (e.g., strings).
            If decode=False, chunk_data is a raw NumPy array.
        """
        with h5py.File(self._path2file, "r") as h5f:
            dset = h5f[path2entry]

            if dset.shape == ():
                raw = dset[()]
                decoded = self._decode_data(raw) if decode else raw
                yield 0, 1, decoded
                return

            length = dset.shape[0]

            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                raw_chunk = dset[start:end]

                if decode:
                    chunk = self._decode_data(raw_chunk)
                else:
                    chunk = np.asarray(raw_chunk)

                yield start, end, chunk

    @staticmethod
    def _read_node(node: Any) -> Any:
        if isinstance(node, h5py.Dataset):
            return H5Reader._decode_data(node[()])

        if isinstance(node, h5py.Group):
            return {name: H5Reader._read_node(node[name]) for name in node.keys()}

        return node

    @staticmethod
    def _decode_data(data: Any) -> Any:
        """Decode HDF5 data into Python-friendly types.

        Recursively decodes NumPy arrays, byte strings, and NumPy scalars.
        Byte values are converted to strings, scalar arrays are unpacked.
        Other values are returned unchanged."""
        if isinstance(data, np.ndarray):
            if data.shape == ():
                return H5Reader._decode_data(data[()])

            if data.dtype.kind in {"S", "O"}:
                # dtype "S" = Byte String Array
                # dtype "O" = Object Array
                return np.array(
                    [H5Reader._decode_data(item) for item in data],
                    dtype=object,
                )
            return np.asarray(data)

        if isinstance(data, bytes):
            return data.decode("utf-8")

        if isinstance(data, np.bytes_):
            return data.decode()

        if isinstance(data, np.generic):
            return data.item()

        return data

    def _build_lazy_tree(self, node, decode: bool, chunk_size: int) -> Any:
        """Recursively build a lazy tree where datasets become callables."""
        if isinstance(node, h5py.Group):
            return {key: self._build_lazy_tree(node[key], decode, chunk_size) for key in node}
        if isinstance(node, h5py.Dataset):
            name = node.name
            # lambda creates a fresh generator when called
            return lambda: self.iter_rows(name, chunk_size=chunk_size, decode=decode)
        return node
