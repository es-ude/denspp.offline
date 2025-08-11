from unittest import TestCase, main
import numpy as np
from denspp.offline.plot_helper import scale_auto_value, translate_unit_to_scale_value, extract_minmax_for_logarithmic_limits


class TestPlots(TestCase):
    input = [1e-14, 1.2e-14, 5.4e-12, 5.6e-10, 9.7e-8, 5.3e-5, 1.1e-2, 1.1e-1, 5.2e0, 4.5e2, 8.3e4, 9.8e6, 1.6e8]
    result0 = [scale_auto_value(val)[0] for val in input]
    result1 = [scale_auto_value(val)[1] for val in input]

    def test_scaling_value(self):
        check = [1e15, 1e15, 1e12, 1e12, 1e9, 1e6, 1e3, 1e3, 1e0, 1e-3, 1e-3, 1e-6, 1e-9]
        np.testing.assert_allclose(np.array(self.result0), np.array(check))

    def test_scaling_unit(self):
        check = ['f', 'f', 'p', 'p', 'n', 'µ', 'm', 'm', '', 'k', 'k', 'M', 'G']
        self.assertEqual("".join(self.result1), "".join(check))

    def test_translate_unit_to_scale_value(self):
        input = {'T': 0, 'M': 0, 'km': 0, ' V': 0, 'mA': 0, 'uV': 0, 'µV': 0, 'nF': 0, 'pC': 0, 'fA': 0, ' fA': 1}
        check = [1e12, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-6, 1e-9, 1e-12, 1e-15, 1e-15]
        result = [translate_unit_to_scale_value(val, pos) for val, pos in input.items()]
        self.assertEqual(result, check)

    def test_extract_minmax_for_logarithmic_limits(self):
        input = np.array([[1e-3, 1e-9, -1e-2, 1.035], [-0.00154, 1e-4, 1e-9, -1e-6]])
        rslt = list()
        for val in input:
            rslt.append(extract_minmax_for_logarithmic_limits(val))

        chck = [[1e-8, 10.], [1e-8, 0.0001]]
        np.testing.assert_array_equal(chck, rslt)

if __name__ == '__main__':
    main()
