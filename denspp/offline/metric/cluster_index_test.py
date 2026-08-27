from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_blobs

from denspp.offline.metric import cluster_index

from .cluster_index import (
    calculate_dunn_index,
    calculate_euclidean_distance,
    calculate_harabasz,
    calculate_silhouette,
    calinski_harabasz_score,
)


class ClusteringMetrics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    def test_euclidean_distance(self):
        rslt = calculate_euclidean_distance(point1=self.X[0], point2=self.X[0])
        self.assertEqual(rslt, 0.0)

    def test_harabasz(self):
        rslt = calculate_harabasz(data=self.X, labels=self.y)
        self.assertGreater(rslt, 1000.0)

    def test_harabasz_score(self):
        rslt = calinski_harabasz_score(X=self.X, labels=self.y)
        self.assertGreater(rslt, 1000.0)

    def test_silhouette(self):
        rslt = calculate_silhouette(data=self.X, labels=self.y)
        self.assertGreater(rslt, 0.7)

    def test_dunn_index(self):
        rslt = calculate_dunn_index(data=self.X, labels=self.y)
        self.assertGreater(rslt, 1.2)

    @patch("denspp.offline.metric.cluster_index.calculate_silhouette")
    def test_returns_difference_between_silhouette_scores(self, mock_calculate_silhouette):
        frames_original = np.array([[0.0, 0.0], [1.0, 1.0]])
        frames_embedding = np.array([[0.0], [1.0]])
        labels = np.array([0, 1])

        mock_calculate_silhouette.side_effect = [0.7, 0.5]

        result = cluster_index.calculate_silhouette_preservation_error(
            frames_original, frames_embedding, labels
        )

        self.assertAlmostEqual(result, 0.2)
        self.assertEqual(mock_calculate_silhouette.call_count, 2)

    @patch("denspp.offline.metric.cluster_index.calculate_silhouette")
    def test_passes_correct_data_and_labels(self, mock_calculate_silhouette):
        frames_original = np.array([[0.0, 0.0], [1.0, 1.0]])
        frames_embedding = np.array([[0.0], [1.0]])
        labels = np.array([0, 1])

        mock_calculate_silhouette.side_effect = [0.7, 0.5]

        cluster_index.calculate_silhouette_preservation_error(frames_original, frames_embedding, labels)

        embedding_call, original_call = mock_calculate_silhouette.call_args_list

        np.testing.assert_array_equal(embedding_call.args[0], frames_embedding)
        np.testing.assert_array_equal(embedding_call.args[1], labels)
        np.testing.assert_array_equal(original_call.args[0], frames_original)
        np.testing.assert_array_equal(original_call.args[1], labels)

    @patch("denspp.offline.metric.cluster_index.calculate_silhouette")
    def test_returns_zero_for_equal_silhouette_scores(self, mock_calculate_silhouette):
        mock_calculate_silhouette.side_effect = [0.5, 0.5]

        result = cluster_index.calculate_silhouette_preservation_error(self.X, self.X, self.y)

        self.assertEqual(result, 0.0)

    @patch("denspp.offline.metric.cluster_index.calculate_silhouette")
    def test_returns_negative_difference(self, mock_calculate_silhouette):
        mock_calculate_silhouette.side_effect = [0.4, 0.7]

        result = cluster_index.calculate_silhouette_preservation_error(self.X, self.X, self.y)

        self.assertAlmostEqual(result, 0.3)

    @patch("denspp.offline.metric.cluster_index.calculate_silhouette")
    def test_uses_same_labels_for_both_spaces(self, mock_calculate_silhouette):
        labels = np.array([0, 0, 1, 1])
        frames_original = np.zeros((4, 2))
        frames_embedding = np.zeros((4, 1))

        mock_calculate_silhouette.side_effect = [0.6, 0.5]

        cluster_index.calculate_silhouette_preservation_error(frames_original, frames_embedding, labels)

        for call in mock_calculate_silhouette.call_args_list:
            np.testing.assert_array_equal(call.args[1], labels)


if __name__ == "__main__":
    main()
