import unittest
import numpy as np
from copy import deepcopy
from .cluster import SettingsCluster, DefaultSettingsCluster, Clustering


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.set0: SettingsCluster = deepcopy(DefaultSettingsCluster)

    def test_kmeans(self):
        set0: SettingsCluster = deepcopy(self.set0)
        set0.type = "kmeans"
        dut = Clustering(set0)
        data = dut.create_dummy_data(
            num_samples=100,
            noise_std=0.1
        )

        dut.init(
            features=data[0],
            true_label=data[1],
        )
        pred = dut.predict(
            features=data[0],
        )
        self.assertEqual(data[1].size, pred.size)

    def test_gmm(self):
        set0: SettingsCluster = deepcopy(self.set0)
        set0.type = "gmm"
        dut = Clustering(set0)
        data = dut.create_dummy_data(
            num_samples=100,
            noise_std=0.1
        )

        dut.init(
            features=data[0],
            true_label=data[1],
        )
        pred = dut.predict(
            features=data[0],
        )
        self.assertEqual(data[1].size, pred.size)

    def test_dbscan(self):
        set0: SettingsCluster = deepcopy(self.set0)
        set0.type = "dbscan"
        dut = Clustering(set0)
        data = dut.create_dummy_data(
            num_samples=100,
            noise_std=0.1
        )

        dut.init(
            features=data[0],
            true_label=data[1],
        )
        pred = dut.predict(
            features=data[0],
        )
        self.assertEqual(data[1].size, pred.size)

    def test_knn(self):
        set0: SettingsCluster = deepcopy(self.set0)
        set0.type = "knn"
        dut = Clustering(set0)
        data = dut.create_dummy_data(
            num_samples=100,
            noise_std=0.1
        )

        dut.init(
            features=data[0],
            true_label=data[1],
        )
        pred = dut.predict(
            features=data[0],
        )
        self.assertEqual(data[1].size, pred.size)


if __name__ == '__main__':
    unittest.main()
