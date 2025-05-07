import numpy as np
from unittest import TestCase, main

# --- Info: Function have to start with test_*
class TestSum(TestCase):
    method = sum
    input = [1, 2, 3, 4]
    result = method(input)

    def test_result_value(self):
        self.assertEqual(self.result, 10)

    def test_result_type(self):
        self.assertEqual(type(self.result), type(int(0)))

    def test_boolean(self):
        self.assertTrue(True)

    def test_numpy(self):
        check = np.array(self.result)
        np.testing.assert_equal(np.sum(self.input),  check)


if __name__ == '__main__':
    main()
