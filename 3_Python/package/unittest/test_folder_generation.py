import unittest
from os import getcwd
from os.path import exists, join
from package.structure_builder import _create_folder_general_firstrun, _create_folder_dnn_firstrun


folder_general = ['data', 'runs', 'test', 'config']
folder_dnn = ['models', 'dataset', 'config']


class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_check_folder_general(self):
        _create_folder_general_firstrun()
        folder2search = '3_Python'
        path2start = join(getcwd().split(folder2search)[0], folder2search)

        num_pass = 0
        for folder in folder_general:
            path2test = join(path2start, folder)
            num_pass += 1 if exists(path2test) else 0
        self.assertEqual(num_pass == len(folder_general), True, "Folders not there")

    def test_check_folder_dnn(self):
        _create_folder_dnn_firstrun()
        folder2search = '3_Python'
        path2start = join(getcwd().split(folder2search)[0], folder2search)

        num_pass = 0
        for folder in folder_dnn:
            path2test = join(path2start, folder)
            num_pass += 1 if exists(path2test) else 0
        self.assertEqual(num_pass == len(folder_dnn), True, "Folders not there")


if __name__ == '__main__':
    unittest.main()
