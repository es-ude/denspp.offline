import unittest

import numpy as np

from denspp.offline.metric.data_numpy import (
    calculate_error_mae,
    calculate_error_mape,
    calculate_error_mbe,
    calculate_error_mpe,
    calculate_error_mse,
    calculate_error_rae,
    calculate_error_rmse,
    calculate_error_rmsre,
    calculate_error_rrmse,
    calculate_error_rse,
)


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
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 0.28133972977262783)

    def test_metric_mpe_float(self):
        rslt = calculate_error_mpe(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 2.0
        self.assertTrue(chck)

    def test_metric_mpe_numpy(self):
        rslt = calculate_error_mpe(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 0.28133972977262783)

    def test_metric_rae_float(self):
        rslt = calculate_error_rae(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 2.0
        self.assertTrue(chck)

    def test_metric_rae_numpy(self):
        rslt = calculate_error_rae(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 0.25)

    def test_metric_rse_float(self):
        rslt = calculate_error_rse(
            y_pred=2.0,
            y_true=-2.0,
        )
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 1.0)

    def test_metric_rse_numpy(self):
        rslt = calculate_error_rse(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 0.0038028169014084506)

    def test_metric_rmse_numpy(self):
        rslt = calculate_error_rmse(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 1.0)

    def test_metric_rrmse_numpy(self):
        rslt = calculate_error_rrmse(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 0.09699312091703453)

    def test_metric_rmsre_numpy(self):
        rslt = calculate_error_rmsre(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        self.assertEqual(type(rslt), float)
        self.assertEqual(rslt, 3.2603112780269354)


if __name__ == "__main__":
    unittest.main()
