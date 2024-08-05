import unittest
from os import remove
from os.path import join

from package.structure_builder import write_data_to_yaml_file, read_yaml_data_to_data

data_wr = {
    'Name': 'John Doe',
    'Position': 'DevOps Engineer',
    'Location': 'England',
    'Age': '26',
    'Experience': {'GitHub': 'Software Engineer', 'Google': 'Technical Engineer', 'Linkedin': 'Data Analyst'},
    'Languages': {'Markup': ['HTML'], 'Programming': ['Python', 'JavaScript', 'Golang']}
}


class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_yaml(self):
        path2yaml = ''
        filename = 'test_output'

        write_data_to_yaml_file(data_wr, filename, path2yaml)
        data_rd = read_yaml_data_to_data(filename, path2yaml)
        self.assertEqual(data_wr == data_rd, True, "Should be equal")

        # --- Checking
        if data_wr == data_rd:
            remove(join(path2yaml, f"{filename}.yaml"))


if __name__ == '__main__':
    unittest.main()
