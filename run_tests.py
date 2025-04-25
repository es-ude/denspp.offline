import unittest
import logging
from shutil import rmtree


if __name__ == '__main__':
    logger = logging.basicConfig(
        level=logging.DEBUG,
        filename='run_test_report.log',
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('denspp', pattern='*_test.py')

    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)

    rmtree('temp_test', ignore_errors=True)
