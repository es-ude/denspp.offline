import unittest
from shutil import rmtree
from os import remove
from os.path import join
from denspp.offline import get_path_to_project_start
from denspp.offline.logger import define_logger_testing


if __name__ == '__main__':
    logger = define_logger_testing()
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('denspp', pattern='*_test.py')

    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)

    rmtree('temp_test', ignore_errors=True)
    remove(join(get_path_to_project_start(), 'access_cloud.yaml'))
