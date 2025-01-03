from unittest import TestCase, main
import numpy as np
from package.tests.test_helper import generate_reference_array
from package.plot.helper import scale_auto_value


class TestPlots(TestCase):
    input = [1.2e-14, 5.4e-12, 5.6e-10, 9.7e-8, 5.3e-5, 1.1e-2, 1.1e-1, 5.2e0, 4.5e2, 8.3e4, 9.8e6, 1.6e8]
    result0 = [scale_auto_value(val)[0] for val in input]
    result1 = [scale_auto_value(val)[1] for val in input]

    def test_scaling_value(self):
        check = [1e15, 1e12, 1e12, 1e9, 1e6, 1e3, 1e3, 1e0, 1e-3, 1e-3, 1e-6, 1e-9]
        np.testing.assert_allclose(np.array(self.result0), np.array(check))

    def test_scaling_unit(self):
        check = ['f', 'p', 'p', 'n', 'µ', 'm', 'm', '', 'k', 'k', 'M', 'G']
        self.assertEqual("".join(self.result1), "".join(check))


if __name__ == '__main__':
    main()
