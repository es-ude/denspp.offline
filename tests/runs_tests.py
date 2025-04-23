import unittest
import logging

if __name__ == '__main__':
    logger = logging.basicConfig(level=logging.DEBUG, filename='debug_test.log')

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.', pattern='test_*.py')

    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)
