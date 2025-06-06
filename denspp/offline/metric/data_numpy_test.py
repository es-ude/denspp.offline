import unittest
from denspp.offline.metric.data_numpy import *


class TestErrorNumpy(unittest.TestCase):
    def test_metric_mbe_float(self):
        rslt = calculate_error_mbe(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 4.0
        self.assertTrue(chck)

    def test_metric_mbe_numpy(self):
        rslt = calculate_error_mbe(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == -1.0
        self.assertTrue(chck)

    def test_metric_mae_float(self):
        rslt = calculate_error_mae(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 4.0
        self.assertTrue(chck)

    def test_metric_mae_numpy(self):
        rslt = calculate_error_mae(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == 1.0
        self.assertTrue(chck)

    def test_metric_mse_float(self):
        rslt = calculate_error_mse(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 16.0
        self.assertTrue(chck)

    def test_metric_mse_numpy(self):
        rslt = calculate_error_mse(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == 1.0
        self.assertTrue(chck)

    def test_metric_mape_float(self):
        rslt = calculate_error_mape(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 2.0
        self.assertTrue(chck)

    def test_metric_mape_numpy(self):
        rslt = calculate_error_mape(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == 0.28133972977262783
        self.assertTrue(chck)


if __name__ == "__main__":
    unittest.main()
