import unittest
from os import getcwd
from os.path import exists, join
from shutil import rmtree
from denspp.offline.structure_builder import init_project_folder, init_dnn_folder


class TestSum(unittest.TestCase):
    folder_general = ['config', 'data', 'runs', 'src_neuro', 'temp']
    folder_dnn = ['models', 'dataset']
    folder2search = 'denspp.offline'
    folder_name_test = 'temp_tests'

    def test_check_folder_general(self):
        init_project_folder(self.folder_name_test)
        path2start = join(getcwd().split(self.folder2search)[0], self.folder2search, self.folder_name_test)

        num_pass = 0
        for folder in self.folder_general:
            path2test = join(path2start, folder)
            num_pass += 1 if exists(path2test) else 0

        rmtree(path2start)
        self.assertEqual(num_pass, len(self.folder_general))

    def test_check_folder_dnn(self):
        init_dnn_folder(self.folder_name_test)
        path2start = join(getcwd().split(self.folder2search)[0], self.folder2search, self.folder_name_test, 'src_dnn')

        num_pass = 0
        for folder in self.folder_dnn:
            path2test = join(path2start, folder)
            num_pass += 1 if exists(path2test) else 0

        rmtree(path2start)
        self.assertEqual(num_pass == len(self.folder_dnn), True, "Folders not there")


if __name__ == '__main__':
    unittest.main()
