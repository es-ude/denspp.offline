import unittest
import numpy as np
from copy import deepcopy
from denspp.offline.dnn import SettingsDataset, DefaultSettingsDataset
from denspp.offline.template.call_dataset import DatasetLoader
from .data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sets: SettingsDataset = deepcopy(DefaultSettingsDataset)
        sets.data_type = 'mnist'
        cls.data = DatasetLoader(
            settings=sets,
            temp_folder=''
        ).load_dataset()

        sets.data_type = 'waveforms'
        cls.data1 = DatasetLoader(
            settings=sets,
            temp_folder=''
        ).load_dataset()

    def setUp(self):
        self.sets: SettingsDataset = deepcopy(DefaultSettingsDataset)

    def test_init(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = ''
        DataProcessor(settings=sets)

    def test_reconfigure_mnist_original(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        rslt = DataProcessor(settings=sets).reconfigure_cluster_with_cell_lib(
            sel_mode_classes=0,
            dataset=self.data
        )

        self.assertEqual(self.data.label.size, rslt.label.size)
        np.testing.assert_array_equal(self.data.label, rslt.label)
        self.assertEqual(self.data.dict, rslt.dict)

    def test_reconfigure_mnist_reduced(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        rslt = DataProcessor(settings=sets).reconfigure_cluster_with_cell_lib(
            sel_mode_classes=1,
            dataset=self.data
        )

        self.assertEqual(self.data.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 2)
        self.assertEqual(rslt.dict, ['even', 'odd'])

    def test_reconfigure_mnist_type(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        rslt = DataProcessor(settings=sets).reconfigure_cluster_with_cell_lib(
            sel_mode_classes=2,
            dataset=self.data
        )

        self.assertEqual(self.data.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 2)
        self.assertEqual(rslt.dict, ['even', 'odd'])

    def test_reconfigure_group(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        rslt = DataProcessor(settings=sets).reconfigure_cluster_with_cell_lib(
            sel_mode_classes=3,
            dataset=self.data
        )

        self.assertEqual(self.data.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 2)
        self.assertEqual(rslt.dict, ['even', 'odd'])

    def test_exclude_none(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        sets.exclude_cluster = []
        rslt = DataProcessor(settings=sets).exclude_cluster_from_dataset(dataset=self.data)

        self.assertEqual(self.data.label.size, rslt.label.size)
        np.testing.assert_array_equal(self.data.label, rslt.label)
        self.assertEqual(self.data.dict, rslt.dict)

    def test_exclude_single(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        sets.exclude_cluster = [1]
        num_missed_labels = np.argwhere(self.data.label == sets.exclude_cluster[0]).flatten().size
        rslt = DataProcessor(settings=sets).exclude_cluster_from_dataset(dataset=self.data)

        self.assertEqual(self.data.label.shape[0] - num_missed_labels, rslt.label.shape[0])
        np.testing.assert_array_equal(self.data.label.size - num_missed_labels, rslt.label.size)
        self.assertEqual(rslt.dict, ['zero', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])

    def test_exclude_dual(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        sets.exclude_cluster = [1, 8]
        num_missed_labels = 0
        for id0 in sets.exclude_cluster:
            num_missed_labels += np.argwhere(self.data.label == id0).flatten().size
        rslt = DataProcessor(settings=sets).exclude_cluster_from_dataset(dataset=self.data)

        self.assertEqual(self.data.label.shape[0]-num_missed_labels, rslt.label.shape[0])
        np.testing.assert_array_equal(self.data.label.size-num_missed_labels, rslt.label.size)
        self.assertEqual(rslt.dict, ['zero', 'two', 'three', 'four', 'five', 'six', 'seven', 'nine'])

    def test_process_vision_normal(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        rslt = DataProcessor(settings=sets).process_vision_datasets(data=self.data)

        self.assertEqual(self.data.data.max(), 255.)
        self.assertEqual(self.data.data.min(), 0.)
        self.assertEqual(self.data.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 10)
        self.assertEqual(self.data.dict, rslt.dict)

    def test_process_vision_normalize(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        sets.normalization_do = True
        rslt = DataProcessor(settings=sets).process_vision_datasets(data=self.data)

        self.assertEqual(rslt.data.max(), 1.)
        self.assertEqual(rslt.data.min(), 0.)
        self.assertEqual(self.data.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 10)
        self.assertEqual(self.data.dict, rslt.dict)

    def test_process_vision_exclude(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        sets.normalization_do = True
        sets.exclude_cluster = [1]
        rslt = DataProcessor(settings=sets).process_vision_datasets(data=self.data)
        num_missed_labels = np.argwhere(self.data.label == sets.exclude_cluster[0]).flatten().size

        self.assertEqual(rslt.data.max(), 1.0)
        self.assertEqual(rslt.data.min(), 0.0)
        self.assertEqual(self.data.label.size-num_missed_labels, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 9)

    def test_process_vision_new_grouping(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        sets.normalization_do = True
        sets.use_cell_sort_mode = 1
        rslt = DataProcessor(settings=sets).process_vision_datasets(data=self.data)

        self.assertEqual(rslt.data.max(), 1.)
        self.assertEqual(rslt.data.min(), 0.)
        self.assertEqual(self.data.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 2)
        self.assertEqual(rslt.dict, ["even", "odd"])

    def test_process_vision_exclude_group(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'mnist'
        sets.normalization_do = True
        sets.use_cell_sort_mode = 1
        sets.exclude_cluster = [1]
        rslt = DataProcessor(settings=sets).process_vision_datasets(data=self.data)

        self.assertEqual(rslt.data.max(), 1.)
        self.assertEqual(rslt.data.min(), 0.)
        self.assertEqual(rslt.label.size, 34418)
        self.assertEqual(rslt.label.size, rslt.data.shape[0])
        self.assertEqual(np.unique(rslt.label).size, 1)
        self.assertEqual(rslt.dict, ["even"])

    def test_process_waveform_normal(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'waveform'
        rslt = DataProcessor(settings=sets).process_vision_datasets(data=self.data1)

        self.assertGreater(rslt.data.max(), 1.5)
        self.assertLess(rslt.data.min(), -1.5)
        self.assertEqual(self.data1.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 12)
        self.assertEqual(self.data1.dict, rslt.dict)

    def test_process_waveform_exclude(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'waveform'
        sets.exclude_cluster = [1]
        rslt = DataProcessor(settings=sets).process_timeseries_datasets(data=self.data1)
        num_missed_labels = np.argwhere(self.data1.label == sets.exclude_cluster[0]).flatten().size

        self.assertGreater(rslt.data.max(), 1.5)
        self.assertLess(rslt.data.min(), -1.5)
        self.assertEqual(self.data1.label.size-num_missed_labels, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 11)

    def test_process_waveform_normalize(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'waveform'
        sets.normalization_do = True
        sets.normalization_method = "minmax"
        rslt = DataProcessor(settings=sets).process_timeseries_datasets(data=self.data1)

        self.assertEqual(rslt.data.max(), 1.0)
        self.assertEqual(rslt.data.min(), -1.0)
        self.assertEqual(self.data1.label.size, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 12)

    def test_process_waveform_reduce_samples(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'waveform'
        sets.normalization_do = True
        sets.normalization_method = "minmax"
        sets.reduce_samples_per_cluster_do = True
        sets.reduce_samples_per_cluster_num = 10
        rslt = DataProcessor(settings=sets).process_timeseries_datasets(data=self.data1)

        self.assertEqual(rslt.data.max(), 1.0)
        self.assertEqual(rslt.data.min(), -1.0)
        self.assertEqual(rslt.label.size, np.unique(rslt.label).size * sets.reduce_samples_per_cluster_num)
        self.assertEqual(np.unique(rslt.label).size, 12)

    def test_process_waveform_augment_samples(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'waveform'
        sets.normalization_do = True
        sets.normalization_method = "minmax"
        sets.augmentation_do = True
        sets.augmentation_num = 100
        rslt = DataProcessor(settings=sets).process_timeseries_datasets(data=self.data1)

        self.assertEqual(rslt.data.max(), 1.0)
        self.assertEqual(rslt.data.min(), -1.0)
        self.assertEqual(rslt.label.size, 13200)
        self.assertEqual(np.unique(rslt.label).size, 12)

    def test_process_waveform_add_noise(self):
        sets: SettingsDataset = deepcopy(self.sets)
        sets.data_type = 'waveform'
        sets.normalization_do = True
        sets.normalization_method = "minmax"
        rslt = DataProcessor(settings=sets).process_timeseries_datasets(data=self.data1, add_noise_cluster=True)

        self.assertEqual(rslt.data.max(), 1.0)
        self.assertEqual(rslt.data.min(), -1.0)
        self.assertEqual(self.data1.label.size+1000, rslt.label.size)
        self.assertEqual(np.unique(rslt.label).size, 13)


if __name__ == '__main__':
    unittest.main()
