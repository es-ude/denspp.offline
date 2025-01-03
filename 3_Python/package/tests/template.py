from unittest import TestCase, main
import numpy as np
from package.tests.test_helper import generate_reference_array


class TestSum(TestCase):
    method = sum
    input = [1, 2, 3, 4]
    result = method(input)

    def test_result_value(self):
        self.assertEqual(self.result, 10, "Should be 10")

    def test_result_type(self):
        self.assertEqual(type(self.result), type(int(0)), "Type should be integer")


if __name__ == '__main__':
    main()
