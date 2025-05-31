from dataclasses import dataclass
import unittest
from os.path import join, exists
from denspp.offline.yaml_handler import YamlConfigHandler
from denspp.offline.structure_builder import get_path_to_project_start


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

# --- DATA FOR TESTING
path2yaml = join(get_path_to_project_start('temp_test'), 'config')
filename = 'Config_Test'
data_wr = {
    'Name': 'John Doe',
    'Position': 'DevOps Engineer',
    'Location': 'England',
    'Age': '26',
    'Experience': {'GitHub': 'Software Engineer', 'Google': 'Technical Engineer', 'Linkedin': 'Data Analyst'},
    'Languages': {'Markup': ['HTML'], 'Programming': ['Python', 'JavaScript', 'Golang']}
}


class TestSum(unittest.TestCase):
    dummy0 = YamlConfigHandler(
        yaml_template=data_wr,
        path2yaml=path2yaml,
        yaml_name=filename + '0'
    )
    dummy1 = YamlConfigHandler(
        yaml_template=DefaultSettingsTest,
        path2yaml=path2yaml,
        yaml_name=filename + '1'
    )

    def test_yaml_create(self):
        self.dummy0.write_dict_to_yaml(data_wr)
        path2chck = join(path2yaml, f"{filename}0.yaml")
        self.assertTrue(exists(path2chck))

    def test_yaml_class(self):
        class_out = self.dummy1.get_class(SettingsTest)
        self.assertTrue(DefaultSettingsTest == class_out)

    def test_yaml_read(self):
        data_rd = self.dummy0.read_yaml_to_dict()
        self.assertTrue(data_wr == data_rd)


if __name__ == '__main__':
    unittest.main()
