import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from denspp.offline.data_call.h5_reader import H5Reader


class H5ReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir.name)
        self.path2file = self.temp_dir / "sample.h5"
        self._write_sample_file()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _write_sample_file(self) -> None:
        with h5py.File(self.path2file, "w") as h5f:
            # Nested group
            nested = h5f.create_group("nested")
            nested.create_dataset("value", data=np.array([1, 2, 3], dtype=np.int32))
            nested.create_dataset("text", data=np.array([b"alpha", b"beta"], dtype="S"))

            # Scalar dataset
            h5f.create_dataset("scalar", data=np.array(7, dtype=np.int32))

            # 1D vector for slicing/streaming tests
            h5f.create_dataset("vector", data=np.arange(10, dtype=np.float32))

            # 2D matrix
            h5f.create_dataset("matrix", data=np.arange(20).reshape(4, 5))

            # Byte string dataset
            h5f.create_dataset("bytes_data", data=np.array([b"hello", b"world"], dtype="S"))

            # Add attributes for show_attrs test
            h5f["vector"].attrs["unit"] = "m/s"
            h5f["vector"].attrs["description"] = "velocity"

    # ------------------------ read_all tests ------------------------

    def test_read_all_returns_lazy_tree(self):
        reader = H5Reader(self.path2file)
        tree = reader.read_all()

        # Top-level keys
        self.assertIn("nested", tree)
        self.assertIn("scalar", tree)
        self.assertIn("vector", tree)

        # Datasets are callables (lambdas)
        self.assertTrue(callable(tree["vector"]))
        self.assertTrue(callable(tree["bytes_data"]))

        # Groups are dicts, not callables
        self.assertIsInstance(tree["nested"], dict)
        self.assertIn("value", tree["nested"])
        self.assertTrue(callable(tree["nested"]["value"]))

    def test_read_all_lazy_iteration(self):
        reader = H5Reader(self.path2file)
        tree = reader.read_all()

        # Get generator from lambda
        gen = tree["vector"]()
        chunks = list(gen)
        self.assertEqual(len(chunks), 1)  # default chunk_size=1000 > 10 rows
        start, end, data = chunks[0]
        self.assertEqual(start, 0)
        self.assertEqual(end, 10)
        np.testing.assert_array_equal(data, np.arange(10, dtype=np.float32))

        # Scalar dataset: still stored as a lambda, but iter_rows handles it
        gen_scalar = tree["scalar"]()
        chunks_scalar = list(gen_scalar)
        self.assertEqual(len(chunks_scalar), 1)
        start, end, data = chunks_scalar[0]
        self.assertEqual(data, 7)

    def test_read_all_with_custom_chunk_size(self):
        reader = H5Reader(self.path2file)
        tree = reader.read_all(chunk_size=3)
        gen = tree["vector"]()
        chunks = list(gen)

        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0][2].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(chunks[-1][2].tolist(), [9.0])

    # ------------------------ read_entry tests ------------------------

    def test_read_entry_streaming_default(self):
        reader = H5Reader(self.path2file)
        gen = reader.read_entry("vector")
        self.assertTrue(hasattr(gen, "__iter__"))  # generator
        chunks = list(gen)
        self.assertEqual(len(chunks), 1)
        start, end, data = chunks[0]
        np.testing.assert_array_equal(data, np.arange(10, dtype=np.float32))

    def test_read_entry_streaming_with_chunk_size(self):
        reader = H5Reader(self.path2file)
        gen = reader.read_entry("vector", chunk_size=3)
        chunks = list(gen)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0][2].tolist(), [0.0, 1.0, 2.0])

    def test_read_entry_streaming_on_scalar(self):
        reader = H5Reader(self.path2file)
        gen = reader.read_entry("scalar")
        chunks = list(gen)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][2], 7)

    def test_read_entry_streaming_errors_on_slicing(self):
        reader = H5Reader(self.path2file)
        with self.assertRaises(ValueError):
            reader.read_entry("vector", start=2, stream=True)

    def test_read_entry_non_streaming_loads_slice(self):
        reader = H5Reader(self.path2file)
        data = reader.read_entry("vector", start=2, stop=8, step=2, stream=False)
        np.testing.assert_array_equal(data, np.array([2.0, 4.0, 6.0], dtype=np.float32))

    def test_read_entry_non_streaming_loads_whole(self):
        reader = H5Reader(self.path2file)
        data = reader.read_entry("vector", stream=False)
        np.testing.assert_array_equal(data, np.arange(10, dtype=np.float32))

    def test_read_entry_non_streaming_group(self):
        reader = H5Reader(self.path2file)
        group = reader.read_entry("nested", stream=False)
        self.assertIsInstance(group, dict)
        self.assertIn("value", group)
        np.testing.assert_array_equal(group["value"], np.array([1, 2, 3]))

    def test_read_entry_non_streaming_scalar(self):
        reader = H5Reader(self.path2file)
        self.assertEqual(reader.read_entry("scalar", stream=False), 7)

    def test_read_entry_non_streaming_slicing_scalar_raises(self):
        reader = H5Reader(self.path2file)
        with self.assertRaises(ValueError):
            reader.read_entry("scalar", start=0, stream=False)

    def test_read_entry_missing_raises_keyerror(self):
        reader = H5Reader(self.path2file)
        with self.assertRaises(KeyError):
            reader.read_entry("nonexistent")

    # ------------------------ show_dataset_structure tests ------------------------

    def test_show_dataset_structure_basic(self):
        reader = H5Reader(self.path2file)
        structure = reader.show_dataset_structure()
        self.assertIn("sample.h5", structure)
        self.assertIn("nested [group]", structure)
        self.assertIn("value [dataset]", structure)
        self.assertIn("scalar [dataset]", structure)
        self.assertIn("vector [dataset]", structure)

    def test_show_dataset_structure_with_max_depth(self):
        reader = H5Reader(self.path2file)
        structure = reader.show_dataset_structure(max_depth=1)
        self.assertIn("nested [group]", structure)
        self.assertNotIn("value [dataset]", structure)  # below depth 1

    def test_show_dataset_structure_with_attrs(self):
        reader = H5Reader(self.path2file)
        structure = reader.show_dataset_structure(show_attrs=True)
        self.assertIn("unit='m/s'", structure)
        self.assertIn("description='velocity'", structure)

    # ------------------------ error cases ------------------------

    def test_fails_for_missing_file(self):
        missing_path = self.temp_dir / "missing.h5"
        with self.assertRaises(FileNotFoundError):
            H5Reader(missing_path)


if __name__ == "__main__":
    unittest.main()
