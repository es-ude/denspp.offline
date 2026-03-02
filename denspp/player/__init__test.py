import unittest
from os.path import join
from denspp.player import (
    get_path_to_project
)


class TestHelpFunction(unittest.TestCase):
    def test_get_path_to_project_wo_ref(self):
        ref = ['denspp', 'offline']
        chck = get_path_to_project()
        rslt = [True for key in ref if key in chck]
        self.assertTrue(sum(rslt) == 2)

    def test_get_path_to_project_with_ref(self):
        ref = ['denspp', 'offline', 'test']
        chck = get_path_to_project(new_folder='test')
        rslt = chck == join(get_path_to_project(), "test")
        self.assertTrue(rslt)


if __name__ == '__main__':
    unittest.main()
