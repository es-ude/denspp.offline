from unittest import TestCase, main
from copy import deepcopy
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset


class TestSettingsDataset(TestCase):
    def test_path2folder0(self):
        set = deepcopy(DefaultSettingsDataset)
        set.data_path = ''
        chck = set.get_path2folder
        self.assertEqual(chck, 'C:\\Git\\denspp.offline\\data\\datasets')

    def test_path2folder1(self):
        set = deepcopy(DefaultSettingsDataset)
        set.data_path = 'C:\\Git\\denspp.offline\\data\\dataset_test'
        chck = set.get_path2folder
        self.assertEqual(chck, 'C:\\Git\\denspp.offline\\data\\dataset_test')

    def test_path2data(self):
        set = deepcopy(DefaultSettingsDataset)
        set.data_path = ''
        chck = set.get_path2data
        self.assertEqual(chck, 'C:\\Git\\denspp.offline\\data\\datasets\\mnist')


if __name__ == '__main__':
    main()
