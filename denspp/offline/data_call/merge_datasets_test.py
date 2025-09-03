import unittest
from denspp.offline.pipeline import PipelineLibrary, DataloaderLibrary
from denspp.offline.data_call import DefaultSettingsData
from .merge_datasets import MergeDataset, SortDataset


class MergeDatasetTest(unittest.TestCase):
    def setUp(self):
        pipe = PipelineLibrary().get_registry("denspp.offline.template").build("PipelineV0", fs_ana=20e3)
        data = DataloaderLibrary().get_registry("denspp.offline.template").build("DataLoader", settings=DefaultSettingsData)

        self.dut = MergeDataset(
            pipeline=pipe,
            dataloader=data,
            settings_data=DefaultSettingsData,
            do_list=False
        )

    def test_access(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
