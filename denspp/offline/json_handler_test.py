from os.path import exists, join
from dataclasses import dataclass
from pathlib import Path
from unittest import TestCase, main
from denspp.offline import get_repo_name, get_path_to_project
from denspp.offline.json_handler import JsonHandler


data_wr = {
    'Name': 'John Doe',
    'Position': 'DevOps Engineer',
    'Location': 'England',
    'Age': '26',
    'Experience': {'GitHub': 'Software Engineer', 'Google': 'Technical Engineer', 'Linkedin': 'Data Analyst'},
    'Languages': {'Markup': ['HTML'], 'Programming': ['Python', 'JavaScript', 'Golang']}
}


@dataclass
class SettingsTest:
    path: str
    val: int
    freq: float
    data: list
    meta: dict

DefaultSettingsTest = SettingsTest(
    path='test',
    val=1,
    freq=10.0,
    data=[0, 1, 2],
    meta={1: 'company', 2: 'street', 3: 'city'}
)


# --- Info: Function have to start with test_*
class TestJSON(TestCase):
    path = join(get_path_to_project('temp_test'), 'config')

    def test_build_file_exists(self):
        JsonHandler(
            template=data_wr,
            path=self.path,
            file_name='test0.json'
        )
        chck = exists(join(self.path, 'test0.json'))
        self.assertTrue(chck)

    def test_build_file_chck_content(self):
        data_rd = JsonHandler(
            template=data_wr,
            path=self.path,
            file_name='test0.json'
        ).get_dict()
        self.assertTrue(data_rd == data_wr)

    def test_build_class_type(self):
        class_rd = JsonHandler(
            template=DefaultSettingsTest,
            path=self.path,
            file_name='test1.json'
        ).get_class(SettingsTest)
        self.assertTrue(type(class_rd) == type(DefaultSettingsTest))

    def test_build_class_content(self):
        class_rd = JsonHandler(
            template=DefaultSettingsTest,
            path=self.path,
            file_name='test1.json'
        ).get_class(SettingsTest)
        self.assertTrue(class_rd.path == DefaultSettingsTest.path)


if __name__ == '__main__':
    main()
