import unittest
from shutil import rmtree
from denspp.offline.logger import define_logger_testing


def load_tests(loader, standard_tests, pattern):
    testcase = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(testcase.discover(start_dir='denspp', pattern='*_test.py'))
    return suite


if __name__ == '__main__':
    logger = define_logger_testing()
    unittest.main()
    rmtree('temp_test', ignore_errors=True)
