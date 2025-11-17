from unittest import main, TestCase
from sklearn.datasets import make_blobs
from .cluster_index import (
    calculate_euclidean_distance,
    calculate_harabasz,
    calculate_silhouette,
    calculate_dunn_index,
    calinski_harabasz_score
)

class ClusteringMetrics(TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    main()
