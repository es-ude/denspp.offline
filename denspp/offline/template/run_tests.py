import unittest
from shutil import rmtree
from denspp.offline.logger import define_logger_testing


if __name__ == '__main__':
    logger = define_logger_testing()
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('src', pattern='*_test.py')
    test_suite.addTest(test_loader.discover('src_dnn', pattern='*_test.py'))
    test_suite.addTest(test_loader.discover('src_pipe', pattern='*_test.py'))

    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)

    rmtree('temp_test', ignore_errors=True)
