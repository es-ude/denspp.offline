from dataclasses import dataclass
import unittest
from os.path import join, exists
from denspp.offline.yaml_handler import YamlHandler
from denspp.offline import get_path_to_project_start


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

data_wr = {
    'Name': 'John Doe',
    'Position': 'DevOps Engineer',
    'Location': 'England',
    'Age': '26',
    'Experience': {'GitHub': 'Software Engineer', 'Google': 'Technical Engineer', 'Linkedin': 'Data Analyst'},
    'Languages': {'Markup': ['HTML'], 'Programming': ['Python', 'JavaScript', 'Golang']}
}


class TestYamlHandler(unittest.TestCase):
    path = join(get_path_to_project_start('temp_test'), 'config')
    file = 'Config_Test'

    dummy0 = YamlHandler(
        template=data_wr,
        path=path,
        file_name=file + '0'
    )
    dummy1 = YamlHandler(
        template=DefaultSettingsTest,
        path=path,
        file_name=file + '1'
    )

    def test_yaml_create_file(self):
        self.dummy0.write_dict_to_yaml(data_wr)
        path2chck = join(self.path, f"{self.file}0.yaml")
        self.assertTrue(exists(path2chck))

    def test_yaml_read_class(self):
        class_out = self.dummy1.get_class(SettingsTest)
        self.assertTrue(DefaultSettingsTest == class_out)

    def test_yaml_read_dict(self):
        data_rd = self.dummy0.get_dict()
        self.assertTrue(data_wr == data_rd)

    def test_yaml_class_type(self):
        class_rd = self.dummy1.get_class(SettingsTest)
        self.assertTrue(type(class_rd) == type(DefaultSettingsTest))


if __name__ == '__main__':
    unittest.main()
