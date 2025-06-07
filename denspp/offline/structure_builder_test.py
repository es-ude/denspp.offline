import unittest
from os import getcwd
from os.path import exists, join
from denspp.offline.structure_builder import init_project_folder, init_dnn_folder
from denspp.offline import (check_key_elements_all, check_key_elements_any,
                            get_repo_name, get_path_to_project_templates, get_path_to_project, get_path_to_project_start)


class TestStructureBuilder(unittest.TestCase):
    folder_general = ['config', 'data', 'dataset', 'runs', 'src', 'src_pipe']
    folder_dnn = ['models', 'dataset']
    folder2search = 'denspp.offline'
    folder_name_test = 'temp_test'

    def test_check_key_elements_all_true(self):
        elements = ['num', 'ber', 'true']
        rslt = check_key_elements_all(
            key='is_the_number_true',
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_key_elements_all_false(self):
        elements = ['num', 'ber', 'false']
        rslt = check_key_elements_all(
            key='is_the_number_true',
            elements=elements
        )
        self.assertFalse(rslt)

    def test_check_key_elements_any_true_one(self):
        elements = ['beatiful', 'ber', 'true']
        rslt = check_key_elements_any(
            key='beatiful_day_today',
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_key_elements_any_true_two(self):
        elements = ['beat', 'iful', 'true']
        rslt = check_key_elements_any(
            key='beatiful_day_today',
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_key_elements_any_true_three(self):
        elements = ['beat', 'iful', 'today']
        rslt = check_key_elements_any(
            key='beatiful_day_today',
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_key_elements_any_false(self):
        elements = ['num', 'ber', 'false']
        rslt = check_key_elements_any(
            key='beatiful_day_today',
            elements=elements
        )
        self.assertFalse(rslt)

    def test_get_repo_name(self):
        ref = ['denspp']
        chck = get_repo_name()
        rslt = True if chck in ref else False
        self.assertTrue(rslt)

    def test_get_path_to_project_wo_ref(self):
        ref = ['denspp', 'offline', 'template']
        chck = get_path_to_project_start()
        rslt = ref[0] in chck and ref[1] in chck
        self.assertTrue(rslt)

    def test_get_path_to_project_with_ref(self):
        chck = get_path_to_project_start(folder_ref='denspp.offline')
        rslt = chck == get_path_to_project_start()
        self.assertTrue(rslt)

    def test_get_path_to_project_app(self):
        chck = get_path_to_project()
        rslt = chck == get_path_to_project_start()
        self.assertTrue(rslt)

    def test_get_path_to_project_templates(self):
        ref = ['denspp', 'offline', 'template']
        chck = get_path_to_project_templates()
        rslt = check_key_elements_all(chck, ref)
        self.assertTrue(rslt)

    def test_check_folder_general(self):
        init_project_folder(self.folder_name_test)
        path2start = join(getcwd().split(self.folder2search)[0], self.folder2search, self.folder_name_test)

        num_pass = 0
        for folder in self.folder_general:
            path2test = join(path2start, folder)
            num_pass += 1 if exists(path2test) else 0

        self.assertEqual(num_pass, len(self.folder_general))

    def test_check_folder_dnn(self):
        init_dnn_folder(self.folder_name_test)
        path2start = join(getcwd().split(self.folder2search)[0], self.folder2search, self.folder_name_test, 'src_dnn')

        num_pass = 0
        for folder in self.folder_dnn:
            path2test = join(path2start, folder)
            num_pass += 1 if exists(path2test) else 0

        self.assertEqual(num_pass == len(self.folder_dnn), True, "Folders not there")


if __name__ == '__main__':
    unittest.main()
