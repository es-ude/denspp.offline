from unittest import main, TestCase
from sklearn.datasets import make_blobs
from .cluster_index import (
    calculate_euclidean_distance,
    calculate_harabasz,
    calinski_harabasz_score,
    calculate_silhouette,
    calculate_dunn_index
)

class ClusteringMetrics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(
            n_samples=100,
            centers=3,
            n_features=2,
            random_state=42
        )

    def test_euclidean_distance(self):
        rslt = calculate_euclidean_distance(
            point1=self.X[0],
            point2=self.X[0]
        )
        self.assertEqual(rslt, 0.)

    def test_harabasz(self):
        rslt = calculate_harabasz(
            data=self.X,
            labels=self.y
        )
        self.assertGreater(rslt, 1000.)

    def test_harabasz_score(self):
        rslt = calinski_harabasz_score(
            X=self.X,
            labels=self.y
        )
        self.assertGreater(rslt, 1000.)

    def test_silhouette(self):
        rslt = calculate_silhouette(
            data=self.X,
            labels=self.y
        )
        self.assertGreater(rslt, 0.7)

    def test_dunn_index(self):
        rslt = calculate_dunn_index(
            data=self.X,
            labels=self.y
        )
        self.assertGreater(rslt, 1.2)


if __name__ == '__main__':
    main()
