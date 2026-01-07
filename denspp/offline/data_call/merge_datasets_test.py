import unittest
from glob import glob
from os.path import join
from copy import deepcopy
from denspp.offline import get_path_to_project
from denspp.offline.pipeline import PipelineLibrary, DataloaderLibrary
from denspp.offline.data_call import DefaultSettingsData
from .merge_datasets import MergeDataset


@unittest.skip("API is broken due to stopped service")
class MergeDatasetTest(unittest.TestCase):
    def setUp(self):
        self.set0 = deepcopy(DefaultSettingsData)
        self.set0.data_set = 'martinez_with_labels'
        self.pipe = PipelineLibrary().get_registry("denspp.offline.template").build_object(
            name="PipelineV0"
        )

    def test_merging_files_with_label_full(self):
        data = DataloaderLibrary().get_registry("denspp.offline.template").build_object(
            name="DataLoaderTest"
        )
        dut = MergeDataset(
            pipeline=self.pipe,
            dataloader=data,
            settings_data=self.set0,
            concatenate_id=False
        )
        dut.get_frames_from_dataset(
            process_points=list(),
            xpos_offset=0
        )
        self.assertEqual(len(glob(join(get_path_to_project('temp_merge'), '*.npy'))), 5)
        dut.merge_data_from_all_iteration()
        self.assertTrue(any(['_Dataset-martinezsimulation_5_Merged' in file_name for file_name in glob(join(get_path_to_project(), 'dataset', '*.npy'))]))

    def test_merging_files_with_label_half(self):
        data = DataloaderLibrary().get_registry("denspp.offline.template").build_object(
            name="DataLoaderTest"
        )
        dut = MergeDataset(
            pipeline=self.pipe,
            dataloader=data,
            settings_data=self.set0,
            concatenate_id=False
        )
        dut.get_frames_from_dataset(
            process_points=[1, 3],
            xpos_offset=0
        )
        self.assertEqual(len(glob(join(get_path_to_project('temp_merge'), '*.npy'))), 2)
        dut.merge_data_from_all_iteration()
        self.assertTrue(any(['_Dataset-martinezsimulation_4_Merged' in file_name for file_name in glob(join(get_path_to_project(), 'dataset', '*.npy'))]))

    def test_merging_files_without_label(self):
        self.set0.data_set = 'martinez_without_labels'
        data = DataloaderLibrary().get_registry("denspp.offline.template").build_object(
            name="DataLoaderTest"
        )
        dut = MergeDataset(
            pipeline=self.pipe,
            dataloader=data,
            settings_data=self.set0,
            concatenate_id=False
        )
        dut.get_frames_from_dataset(
            process_points=[],
            xpos_offset=0
        )
        self.assertEqual(len(glob(join(get_path_to_project('temp_merge'), '*.npy'))), 5)
        dut.merge_data_from_all_iteration()
        self.assertTrue(any(['_Dataset-martinezsimulation_5_Merged' in file_name for file_name in glob(join(get_path_to_project(), 'dataset', '*.npy'))]))


if __name__ == '__main__':
    unittest.main()
