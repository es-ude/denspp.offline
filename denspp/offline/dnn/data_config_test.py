import numpy as np
from copy import deepcopy
from unittest import TestCase, main, skip
from os.path import join
from denspp.offline import get_path_to_project
from denspp.offline.template.call_dataset import DatasetLoader
from .data_config import (
    DatasetFromFile,
    SettingsDataset,
    TransformLabels,
    logic_combination
)


TestSettingsDataset = SettingsDataset(
    data_path='temp_test',
    data_type='',
    use_cell_sort_mode=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=False,
    normalization_method='minmax',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[]
)


class TestDatasetSettings(TestCase):
    def test_main_folder(self):
        set = deepcopy(TestSettingsDataset)
        chck = set.get_path2folder_project
        self.assertEqual(str(chck), get_path_to_project())

    def test_path2folder_default(self):
        set = deepcopy(TestSettingsDataset)
        set.data_path = ''
        chck = set.get_path2folder
        self.assertEqual(str(chck), get_path_to_project())

    def test_path2folder_relative(self):
        set = deepcopy(TestSettingsDataset)
        set.data_path = 'dataset'
        chck = set.get_path2folder
        self.assertEqual(str(chck), join(get_path_to_project(), 'dataset'))

    def test_path2folder_absolute(self):
        set = deepcopy(TestSettingsDataset)
        set.data_path = join(get_path_to_project(), 'temp_test')
        chck = set.get_path2folder
        self.assertEqual(str(chck), join(get_path_to_project(), 'temp_test'))


class TestDatasetConfig(TestCase):
    def setUp(self):
        self.sets: SettingsDataset = deepcopy(TestSettingsDataset)

    @skip
    def test_dataset_overview(self):
        self.sets.data_type = 'mnist'
        overview = DatasetLoader(
            settings=self.sets,
            temp_folder='temp_test'
        ).print_overview_datasets(do_print=False)
        self.assertGreater(len(overview), 4)

    def test_dataset_empty(self):
        self.sets.data_type = ''
        try:
            DatasetLoader(settings=self.sets).load_dataset(do_print=False)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_dataset_mnist(self):
        self.sets.data_type = 'mnist'
        try:
            data = DatasetLoader(settings=self.sets).load_dataset()
            self.assertEqual(type(data), DatasetFromFile)
            self.assertEqual(data.data.shape, (70000, 28, 28))
            self.assertEqual(len(data.dict), 10)
            self.assertEqual(data.label.size, 70000)
        except:
            self.assertTrue(False)

    def test_dataset_waveforms(self):
        self.sets.data_type = 'waveforms'
        try:
            data = DatasetLoader(settings=self.sets).load_dataset()
            self.assertEqual(type(data), DatasetFromFile)
            self.assertEqual(data.data.shape, (12000, 280))
            self.assertEqual(len(data.dict), 12)
            self.assertEqual(data.label.size, 12000)
        except:
            self.assertTrue(False)


class TestLabelTransformation(TestCase):
    def test_logic_combination(self):
        true = np.array((0, 1, 1, 2, 0, 1, 1, 2, 1, 1, 2, 3, 3, 0, 1, 0, 2, 3), dtype=np.uint8)
        pred = np.array((0, 1, 2, 2, 0, 1, 1, 2, 1, 1, 2, 2, 3, 0, 1, 0, 2, 3), dtype=np.uint8)
        tran = [[0, 2], [1, 3]]

        rslt = logic_combination(
            labels_in=TransformLabels(
                true=true,
                pred=pred
            ),
            translate_list=tran
        )
        chck_true = np.array((0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1), dtype=np.uint8)
        chck_pred = np.array((0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1), dtype=np.uint8)
        self.assertTrue(np.array_equal(chck_true, rslt.true) and np.array_equal(chck_pred, rslt.pred))


if __name__ == '__main__':
    main()
