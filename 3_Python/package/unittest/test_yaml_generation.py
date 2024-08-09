import unittest
from os import remove
from os.path import join, exists

from package.yaml_handler import write_dict_to_yaml, read_yaml_to_dict

# --- DATA FOR TESTING
path2yaml = ''
filename = 'test_output'
data_wr = {
    'Name': 'John Doe',
    'Position': 'DevOps Engineer',
    'Location': 'England',
    'Age': '26',
    'Experience': {'GitHub': 'Software Engineer', 'Google': 'Technical Engineer', 'Linkedin': 'Data Analyst'},
    'Languages': {'Markup': ['HTML'], 'Programming': ['Python', 'JavaScript', 'Golang']}
}
path2chck = join(path2yaml, f"{filename}.yaml")


class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_yaml_create(self):
        write_dict_to_yaml(data_wr, filename, path2yaml)
        self.assertEqual(exists(path2chck), True, "YAML file should be there")

    def test_yaml_read(self):
        data_rd = read_yaml_to_dict(filename, path2yaml)
        self.assertEqual(data_wr == data_rd, True, "Should be equal")
        if data_wr == data_rd:
            remove(path2chck)


if __name__ == '__main__':
    unittest.main()
