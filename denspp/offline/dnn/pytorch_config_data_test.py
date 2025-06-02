from unittest import TestCase, main
from copy import deepcopy
from os.path import join
from denspp.offline import get_path_to_project_start
from denspp.offline.dnn.pytorch_config_data import SettingsDataset
from denspp.offline.template.call_dataset import DatasetLoader


TestSettingsDataset = SettingsDataset(
    data_path='temp_test',
    data_file_name='',
    use_cell_sort_mode=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=False,
    normalization_method='minmax',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[]
)


class TestDataset(TestCase):
    def test_path2folder0(self):
        set = deepcopy(TestSettingsDataset)
        set.data_path = ''
        chck = set.get_path2folder
        self.assertEqual(chck, join(get_path_to_project_start(), 'dataset'))

    def test_path2folder1(self):
        set = deepcopy(TestSettingsDataset)
        set.data_path = join(get_path_to_project_start(), 'temp_test')
        chck = set.get_path2folder
        self.assertEqual(chck, join(get_path_to_project_start(), 'temp_test'))

    def test_path2data(self):
        set = deepcopy(TestSettingsDataset)
        set.data_file_name = 'mnist'
        chck = set.get_path2data
        self.assertEqual(chck, join(get_path_to_project_start(), TestSettingsDataset.data_path, 'dataset', 'mnist'))

    def test_dataset_overview(self):
        set = deepcopy(TestSettingsDataset)
        set.data_file_name = 'mnist'
        overview = DatasetLoader(settings=set).print_overview_datasets(do_print=False)
        self.assertGreater(len(overview), 4)

    def test_dataset_empty(self):
        set = deepcopy(TestSettingsDataset)
        set.data_file_name = ''
        try:
            DatasetLoader(settings=set).load_dataset(do_print=False)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_dataset_mnist(self):
        set = deepcopy(TestSettingsDataset)
        set.data_file_name = 'mnist'
        try:
            data = DatasetLoader(settings=set).load_dataset()
            self.assertEqual(data['data'].shape, (70000, 28, 28))
        except:
            self.assertTrue(False)

    def test_dataset_fashion(self):
        set = deepcopy(TestSettingsDataset)
        set.data_file_name = 'fashion'
        try:
            data = DatasetLoader(settings=set).load_dataset()
            self.assertEqual(data['data'].shape, (70000, 28, 28))
        except:
            self.assertTrue(False)

    def test_dataset_cifar10(self):
        set = deepcopy(TestSettingsDataset)
        set.data_file_name = 'cifar10'
        try:
            data = DatasetLoader(settings=set).load_dataset()
            self.assertEqual(data['data'].shape, (60000, 32, 32, 3))
        except:
            self.assertTrue(False)

    def test_dataset_cifar100(self):
        set = deepcopy(TestSettingsDataset)
        set.data_file_name = 'cifar100'
        try:
            data = DatasetLoader(settings=set).load_dataset()
            self.assertEqual(data['data'].shape, (60000, 32, 32, 3))
        except:
            self.assertTrue(False)


if __name__ == '__main__':
    main()
