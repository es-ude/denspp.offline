from unittest import TestCase, main
import numpy as np
import torch
from denspp.offline.preprocessing.frame_normalization import DataNormalization


def generate_test_data(num_samples: int=100, num_window_size: int=32, do_tensor: bool=False) -> np.ndarray | torch.Tensor:
    """Test function for generating test data
    :param num_samples:         Number of samples to generate
    :param num_window_size:     Number of window size
    :param do_tensor:           Boolean if output type is tensor
    :return:                    Tensor or numpy array with test data
    """
    if do_tensor:
        x = torch.linspace(0, 2*torch.pi, num_window_size)
        data = torch.zeros((num_samples, num_window_size))
        for idx in range(num_samples):
            data[idx, :] = 10 * torch.sin(x - idx / num_samples * torch.pi)
    else:
        x = np.linspace(0, 2 * np.pi, num_window_size)
        data = np.zeros((num_samples, num_window_size))
        for idx in range(num_samples):
            data[idx, :] = 10 * np.sin(x - idx / num_samples * np.pi)
    return data


def generate_reference_array(data_array: np.ndarray | list, index_array: str='check') -> str:
    """Function for generating a reference array to load into the testbench
    :param data_array:  numpy array or list with data
    :param index_array: index name of data array for testing
    :return:            string with index name and list array with data
    """
    if type(data_array) == np.ndarray:
        list_to_transfer = data_array.tolist()
    else:
        list_to_transfer = data_array
    list_to_transfer = [str(val) for val in list_to_transfer]

    string_out = f'{index_array} = [' + ', '.join(list_to_transfer) + ']'
    return string_out


class TestHelper(TestCase):
    num_input = 10
    num_window_size = 100

    def test_generate_reference_list_value(self):
        test_data = [idx for idx in range(10)]
        result = generate_reference_array(test_data, 'check')
        local_vars = {}
        exec(result, {}, local_vars)

        np.testing.assert_array_equal(np.array(test_data), np.array(local_vars['check']))

    def test_generate_reference_integer(self):
        test_data = [idx for idx in range(10)]
        result = generate_reference_array(test_data, 'check')
        check = 'check = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
        np.testing.assert_array_equal(result, check)

    def test_generate_reference_float(self):
        test_data = np.linspace(-1, +1, 11, endpoint=True, dtype=float)
        result = generate_reference_array(test_data, 'check')
        check = 'check = [-1.0, -0.8, -0.6, -0.3999999999999999, -0.19999999999999996, 0.0, 0.20000000000000018, 0.40000000000000013, 0.6000000000000001, 0.8, 1.0]'
        np.testing.assert_array_equal(result, check)

    def test_generate_testdata_type_tensor(self):
        test_data = generate_test_data(num_samples=self.num_input, num_window_size=self.num_window_size, do_tensor=True)
        self.assertEqual(test_data.dtype, torch.float32)

    def test_generate_testdata_shape_tensor(self):
        test_data = generate_test_data(num_samples=self.num_input, num_window_size=self.num_window_size, do_tensor=True)
        self.assertEqual(test_data.shape, torch.Size([self.num_input, self.num_window_size]))

    def test_generate_testdata_type_numpy(self):

        test_data = generate_test_data(num_samples=self.num_input, num_window_size=self.num_window_size, do_tensor=False)
        self.assertEqual(test_data.dtype, np.float64)

    def test_generate_testdata_shape_numpy(self):
        test_data = generate_test_data(num_samples=self.num_input, num_window_size=self.num_window_size, do_tensor=False)
        self.assertEqual(test_data.shape, (self.num_input, self.num_window_size))


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
