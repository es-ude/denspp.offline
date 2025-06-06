import unittest
from torch import Tensor, tensor
from denspp.offline.metric.data_torch import (calculate_number_false_predictions, calculate_number_true_predictions,
                                              calculate_precision, calculate_recall, calculate_fbeta)


class TestErrorTorch(unittest.TestCase):
    def test_metric_number_true(self):
        rslt = calculate_number_true_predictions(
            pred=tensor([[1., 2., 3., 4.], [1., 2., 2., 4.]]),
            true=tensor([[1., 2., 3., 4.], [1., 2., 3., 4.]])
        )
        chck = Tensor([7]) == rslt and type(rslt) == Tensor
        self.assertTrue(chck)

    def test_metric_number_false(self):
        rslt = calculate_number_false_predictions(
            pred=tensor([[1., 2., 3., 4.], [1., 2., 2., 4.]]),
            true=tensor([[1., 2., 3., 4.], [1., 2., 3., 4.]])
        )
        chck = Tensor([1]) == rslt and type(rslt) == Tensor
        self.assertTrue(chck)

    def test_metric_precision_value(self):
        rslt = calculate_precision(
            pred=tensor([1., 0., 1., 0., 1., 0., 0., 1.]),
            true=tensor([1., 0., 1., 0., 1., 1., 1., 1.])
        )
        chck = Tensor([0.75])
        self.assertEqual(rslt, chck)

    def test_metric_precision_type(self):
        rslt = calculate_precision(
            pred=tensor([1., 0., 1., 0., 1., 0., 0., 1.]),
            true=tensor([1., 0., 1., 0., 1., 1., 1., 1.])
        )
        self.assertEqual(type(rslt), Tensor)

    def test_metric_recall_value(self):
        rslt = calculate_recall(
            pred=tensor([1., 0., 1., 0., 1., 0., 0., 1.]),
            true=tensor([1., 0., 1., 2., 1., 1., 1., 1.])
        )
        chck = Tensor([0.625])
        self.assertEqual(rslt, chck)

    def test_metric_recall_type(self):
        rslt = calculate_recall(
            pred=tensor([1., 0., 1., 0., 1., 0., 0., 1.]),
            true=tensor([1., 0., 1., 2., 1., 1., 1., 1.])
        )
        self.assertEqual(type(rslt), Tensor)

    def test_metric_fbeta_value(self):
        rslt = calculate_fbeta(
            pred=tensor([1., 0., 1., 0., 1., 0., 0., 1.]),
            true=tensor([1., 0., 1., 1., 1., 1., 1., 1.]),
            beta=1
        )
        chck = Tensor([0.625])
        self.assertEqual(rslt, chck)

    def test_metric_fbeta_type(self):
        rslt = calculate_fbeta(
            pred=tensor([1., 0., 1., 0., 1., 0., 0., 1.]),
            true=tensor([1., 0., 1., 1., 1., 1., 1., 1.]),
            beta=1
        )
        self.assertEqual(type(rslt), Tensor)


if __name__ == "__main__":
    unittest.main()
