import numpy as np
from unittest import TestCase, main


def generate_reference_array(data_array: np.ndarray | list, index_array = 'check') -> str:
    """"""
    if type(data_array) == np.ndarray:
        list_to_transfer = data_array.tolist()
    else:
        list_to_transfer = data_array
    list_to_transfer = [str(val) for val in list_to_transfer]

    string_out = f'{index_array} = [' + ', '.join(list_to_transfer) + ']'
    return string_out


class TestHelper(TestCase):
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


if __name__ == '__main__':
    main()
