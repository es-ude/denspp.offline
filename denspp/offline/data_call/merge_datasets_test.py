import unittest
from glob import glob
from os.path import join
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import h5py
import numpy as np

from denspp.offline.data_call.merge_datasets import MergeDataset
from denspp.offline.preprocessing import FrameWaveform


class MergeDatasetTest(unittest.TestCase):
    def setUp(self):
        self._test_project_dir = TemporaryDirectory()
        self.project_dir = Path(self._test_project_dir.name)

        self.temp_dir = self.project_dir / "temp_merge"
        self.temp_dir.mkdir(exist_ok=True)

        self.dataset_dir = self.project_dir / "dataset"
        self.dataset_dir.mkdir(exist_ok=True)

        self.get_path_patcher = patch(
            "denspp.offline.data_call.merge_datasets.get_path_to_project",
            side_effect=self._get_test_path_to_project,
        )
        self.get_path_patcher.start()

        self.pipeline_mock = Mock()
        self.pipeline_mock.run_preprocessor = Mock()
        self.pipeline_mock.run_classifier = Mock()
        self.pipeline_mock.settings = Mock()
        self.pipeline_mock.run_preprocessor.__name__ = "run_preprocessor"
        self.pipeline_mock.run_classifier.__name__ = "run_classifier"
        self.pipeline_mock.settings.__name__ = "settings"

        self.dataloader_mock = Mock()
        self.settings_mock = Mock()

    def _get_test_path_to_project(self, folder_name):
        return str(self.project_dir / folder_name)

    def _create_dummy_frame(self, waveform, xpos, label, sampling_rate=1000):
        return FrameWaveform(
            waveform=np.array(waveform, dtype=np.float32),
            xpos=np.array([xpos], dtype=np.int32),
            label=np.array([label], dtype=np.int32),
            sampling_rate=sampling_rate,
        )

    def _save_dummy_dataset(self, file_name, frames_list):
        self.temp_dir.mkdir(exist_ok=True)
        path2file = self.temp_dir / file_name

        waveforms = []
        positions = []
        labels = []

        sampling_rate = frames_list[0].sampling_rate if frames_list else 1000

        for frame in frames_list:
            frame_waveforms = np.asarray(frame.waveform, dtype=np.float32)
            if frame_waveforms.ndim == 1:
                frame_waveforms = frame_waveforms[np.newaxis, :]

            frame_positions = np.atleast_1d(np.asarray(frame.xpos, dtype=np.int32))
            frame_labels = np.atleast_1d(np.asarray(frame.label, dtype=np.int32))

            for waveform, xpos, label in zip(frame_waveforms, frame_positions, frame_labels):
                waveforms.append(waveform)
                positions.append(xpos)
                labels.append(label)

        with h5py.File(path2file, "w") as h5f:
            h5f.create_dataset("waveform", data=np.asarray(waveforms, dtype=np.float32))
            h5f.create_dataset("xpos", data=np.asarray(positions, dtype=np.int32))
            h5f.create_dataset("label", data=np.asarray(labels, dtype=np.int32))
            h5f.create_dataset("sampling_rate", data=np.asarray(sampling_rate))

    def test_merge_without_concatenation(self):
        ds1 = [
            self._create_dummy_frame([1, 2, 3], 10, 0),
            self._create_dummy_frame([4, 5, 6], 20, 1),
        ]
        ds2 = [
            self._create_dummy_frame([7, 8, 9], 30, 0),
            self._create_dummy_frame([10, 11, 12], 40, 2),
            self._create_dummy_frame([13, 14, 15], 50, 5),
        ]
        ds3 = [
            self._create_dummy_frame([16, 17, 18], 60, 3),
        ]

        self._save_dummy_dataset("2024-01-01_Dataset-A.h5", ds1)
        self._save_dummy_dataset("2024-01-02_Dataset-B.h5", ds2)
        self._save_dummy_dataset("2024-01-03_Dataset-C.h5", ds3)

        dut = MergeDataset(
            pipeline=self.pipeline_mock,
            dataloader=self.dataloader_mock,
            settings_data=self.settings_mock,
            concatenate_id=False,
        )
        dut.merge_data_from_all_iteration()

        merged_files = sorted(glob(join(self.dataset_dir, "*_Merged.h5")))
        self.assertEqual(len(merged_files), 1)

        with h5py.File(merged_files[0], "r") as h5f:
            merged_data = {
                "data": np.array(h5f["data"][()]),
                "class": np.array(h5f["class"][()]),
                "position": np.array(h5f["position"][()]),
            }

        labels = merged_data["class"]
        expected_labels = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(labels, expected_labels)

    def test_merge_with_concatenation(self):
        ds1 = [
            self._create_dummy_frame([1, 2, 3], 10, 0),
            self._create_dummy_frame([4, 5, 6], 20, 1),
        ]
        ds2 = [
            self._create_dummy_frame([7, 8, 9], 30, 0),
            self._create_dummy_frame([10, 11, 12], 40, 2),
        ]

        self._save_dummy_dataset("2024-01-01_Dataset-A.h5", ds1)
        self._save_dummy_dataset("2024-01-02_Dataset-B.h5", ds2)

        dut = MergeDataset(
            pipeline=self.pipeline_mock,
            dataloader=self.dataloader_mock,
            settings_data=self.settings_mock,
            concatenate_id=True,
        )
        dut.merge_data_from_all_iteration()

        merged_files = sorted(glob(join(self.dataset_dir, "*_Merged.h5")))
        self.assertEqual(len(merged_files), 1)

        with h5py.File(merged_files[0], "r") as h5f:
            merged_data = {
                "data": np.array(h5f["data"][()]),
                "class": np.array(h5f["class"][()]),
                "position": np.array(h5f["position"][()]),
            }

        labels = merged_data["class"]
        expected_labels = np.array([0, 1, 0, 2])
        np.testing.assert_array_equal(labels, expected_labels)

    def test_expected_keys(self):
        expected_keys = {
            "data",
            "class",
            "position",
            "sampling_rate",
            "create_time",
            "label_mapping_keys",
            "label_mapping_values",
        }

        ds1 = [
            self._create_dummy_frame([1, 2, 3], 10, 0),
            self._create_dummy_frame([4, 5, 6], 20, 1),
        ]

        self._save_dummy_dataset("2024-01-01_Dataset-A.h5", ds1)

        dut = MergeDataset(
            pipeline=self.pipeline_mock,
            dataloader=self.dataloader_mock,
            settings_data=self.settings_mock,
            concatenate_id=False,
        )

        dut.merge_data_from_all_iteration()

        merged_files = sorted(glob(join(self.dataset_dir, "*_Merged.h5")))
        self.assertEqual(len(merged_files), 1)

        with h5py.File(merged_files[0], "r") as h5f:
            self.assertEqual(set(h5f.keys()), expected_keys)

    def tearDown(self):
        self.get_path_patcher.stop()
        self._test_project_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
