from unittest import TestCase, main
import numpy as np
from denspp.offline.data_process.frame_normalization import DataNormalization
from tests.test_helper import generate_test_data


class TestSum(TestCase):
    x = np.linspace(start=0, stop=3, num=int(2*np.pi*1000), endpoint=True)
    input_numpy = generate_test_data(do_tensor=False)
    input_torch = generate_test_data(do_tensor=True)

    def test_list_methods(self):
        test_func = DataNormalization(method='minmax', peak_mode=0)
        key = test_func.list_normalization_methods(False)
        self.assertEqual(len(key), 6)

    def test_error_wrong_input(self):
        test_func = DataNormalization(method='bimax', peak_mode=0)
        try:
            data = test_func.normalize(self.input_torch)
            result = False
        except:
            result = True

        self.assertEqual(result, True)

    def test_numpy_minmax_max(self):
        test_func = DataNormalization(method='minmax', peak_mode=0)
        data = test_func.normalize(self.input_numpy)

        result = (data.min(), data.max())
        expected = (-1.0051571362062028, 1.0)
        self.assertEqual(result, expected)

    def test_numpy_minmax_min(self):
        test_func = DataNormalization(method='minmax', peak_mode=1)
        data = test_func.normalize(self.input_numpy)

        result = (data.min(), data.max())
        expected = (-1.0, 1.0050535609440512)
        self.assertEqual(result, expected)

    def test_numpy_minmax_absmax(self):
        test_func = DataNormalization(method='minmax', peak_mode=2)
        data = test_func.normalize(self.input_numpy)

        result = (data.min(), data.max())
        expected = (-1.0, 1.0)
        self.assertEqual(result, expected)

    def test_torch_minmax_max(self):
        test_func = DataNormalization(method='minmax', peak_mode=0)
        data = test_func.normalize(self.input_torch)

        result = (data.min(), data.max())
        expected = (-1.0051571362062028, 1.0)
        self.assertEqual(result, expected)

    def test_torch_minmax_min(self):
        test_func = DataNormalization(method='minmax', peak_mode=1)
        data = test_func.normalize(self.input_torch)

        result = (data.min(), data.max())
        expected = (-1.0, 1.0050535609440512)
        self.assertEqual(result, expected)

    def test_torch_minmax_absmax(self):
        test_func = DataNormalization(method='minmax', peak_mode=2)
        data = test_func.normalize(self.input_torch)

        result = (data.min(), data.max())
        expected = (-1.0, 1.0)
        self.assertEqual(result, expected)

    def test_numpy_zeroone_max(self):
        test_func = DataNormalization(method='zeroone', peak_mode=0)
        data = test_func.normalize(self.input_numpy)

        result = (data.min(), data.max())
        expected = (-0.0025785681031014196, 1.0)
        self.assertEqual(result, expected)

    def test_numpy_zeroone_min(self):
        test_func = DataNormalization(method='zeroone', peak_mode=1)
        data = test_func.normalize(self.input_numpy)

        result = (data.min(), data.max())
        expected = (0.0, 1.0025267804720257)
        self.assertEqual(result, expected)

    def test_numpy_zeroone_absmax(self):
        test_func = DataNormalization(method='zeroone', peak_mode=2)
        data = test_func.normalize(self.input_numpy)

        result = (data.min(), data.max())
        expected = (0.0, 1.0)
        self.assertEqual(result, expected)

    def test_torch_zeroone_max(self):
        test_func = DataNormalization(method='zeroone', peak_mode=0)
        data = test_func.normalize(self.input_torch)

        result = (float(data.min()), float(data.max()))
        expected = (-0.002578556537628174, 1.0)
        self.assertEqual(result, expected)

    def test_torch_zeroone_min(self):
        test_func = DataNormalization(method='zeroone', peak_mode=1)
        data = test_func.normalize(self.input_torch)

        result = (data.min(), data.max())
        expected = (0.0, 1.0025267804720257)
        self.assertEqual(result, expected)

    def test_torch_zeroone_absmax(self):
        test_func = DataNormalization(method='zeroone', peak_mode=2)
        data = test_func.normalize(self.input_torch)

        result = (data.min(), data.max())
        expected = (0.0, 1.0)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    main()
