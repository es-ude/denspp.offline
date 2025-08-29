import unittest
from .timestamps import compare_timestamps, MetricTimestamps


class CompareTimestampsTest(unittest.TestCase):
    def test_simple_list_same(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [4, 8, 20, 42, 80, 102]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true,
            pred_labels=pos_pred,
            window=2
        )
        self.assertEqual(metrics.f1_score, 1.0)

    def test_simple_list_same_inside_window(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [2, 9, 20, 41, 81, 103]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true,
            pred_labels=pos_pred,
            window=2
        )
        self.assertEqual(metrics.f1_score, 1.0)

    def test_simple_list_almost_same(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [2, 9, 20, 45, 77, 105]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true,
            pred_labels=pos_pred,
            window=2
        )
        self.assertEqual(metrics.f1_score, 0.5)

    def test_simple_list_different_size0(self):
        pos_true = [4, 8, 20, 42, 80, 102, 115, 134]
        pos_pred = [4, 8, 20, 42, 80, 102]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true,
            pred_labels=pos_pred,
            window=2
        )
        self.assertEqual(metrics.f1_score, 3/4)

    def test_simple_list_different_size1(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [4, 8, 20, 42, 80, 102, 115, 134]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true,
            pred_labels=pos_pred,
            window=2
        )
        self.assertEqual(metrics.f1_score, 3/4)


if __name__ == '__main__':
    unittest.main()
