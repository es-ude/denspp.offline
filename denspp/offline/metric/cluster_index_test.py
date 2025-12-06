from unittest import main, TestCase
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils.multiclass import unique_labels

from .cluster_index import (
    calculate_euclidean_distance,
    calculate_harabasz,
    calculate_silhouette,
    calculate_dunn_index
)

class EuclideanDistanceTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_dim_2D, _ = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
        cls.x_dim_5D, _ = make_blobs(n_samples=100, centers=2, n_features=5, random_state=42)
        cls.x_dim_10D, _ = make_blobs(n_samples=50, centers=3, n_features=10, random_state=42)

    def test_mismatching_points(self):
        with self.assertRaises(ValueError):
            calculate_euclidean_distance(self.x_dim_2D[0], self.x_dim_5D[1])

        with self.assertRaises(ValueError):
            calculate_euclidean_distance(self.x_dim_2D[0], self.x_dim_10D[1])

        with self.assertRaises(ValueError):
            calculate_euclidean_distance(self.x_dim_5D[0], self.x_dim_10D[1])

    def test_identical_points(self):
        for point_idx in range(10):
            dist = calculate_euclidean_distance(self.x_dim_2D[point_idx], self.x_dim_2D[point_idx])
            self.assertEqual(dist, 0)

    def test_distance_2Ddims(self):
        for pair_idx in range(0, len(self.x_dim_2D) - 1, 2):
            dist = calculate_euclidean_distance(self.x_dim_2D[pair_idx], self.x_dim_2D[pair_idx + 1])
            expected = np.linalg.norm(self.x_dim_2D[pair_idx] - self.x_dim_2D[pair_idx + 1])
            self.assertAlmostEqual(dist, expected)

    def test_distance_5Ddims(self):
        for pair_idx in range(0, len(self.x_dim_5D) - 1, 2):
            dist = calculate_euclidean_distance(self.x_dim_5D[pair_idx], self.x_dim_5D[pair_idx + 1])
            expected = np.linalg.norm(self.x_dim_5D[pair_idx] - self.x_dim_5D[pair_idx + 1])
            self.assertAlmostEqual(dist, expected)

    def test_distance_10Ddims(self):
        for pair_idx in range(0, len(self.x_dim_10D) - 1, 2):
            dist = calculate_euclidean_distance(self.x_dim_10D[pair_idx], self.x_dim_10D[pair_idx + 1])
            expected = np.linalg.norm(self.x_dim_10D[pair_idx] - self.x_dim_10D[pair_idx + 1])
            self.assertAlmostEqual(dist, expected)


class SetUpBlob(TestCase):
    @classmethod
    def setUpClass(cls):
        # 2D
        cls.x_dim2D_close_cl_low_std, cls.labels_dim2D_close_cl_low_std = make_blobs(
            n_samples=100, centers=[[0, 0], [1, 1]], n_features=2, cluster_std=0.2, random_state=42)
        cls.x_dim2D_close_cl_high_std, cls.labels_dim2D_close_cl_high_std = make_blobs(
            n_samples=100, centers=[[0, 0], [1, 1]], n_features=2, cluster_std=2.0, random_state=42)
        cls.x_dim2D_distant_cl_low_std, cls.labels_dim2D_distant_cl_low_std = make_blobs(
            n_samples=100, centers=[[0, 0], [10, 10]], n_features=2, cluster_std=0.2, random_state=42)
        cls.x_dim2D_distant_cl_high_std, cls.labels_dim2D_distant_cl_high_std = make_blobs(
            n_samples=100, centers=[[0, 0], [10, 10]], n_features=2, cluster_std=2.0, random_state=42)

        # 5D
        cls.x_dim5D_close_cl_low_std, cls.labels_dim5D_close_cl_low_std = make_blobs(
            n_samples=100, centers=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
            n_features=5, cluster_std=0.2, random_state=42)
        cls.x_dim5D_close_cl_high_std, cls.labels_dim5D_close_cl_high_std = make_blobs(
            n_samples=100, centers=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
            n_features=5, cluster_std=2.0, random_state=42)
        cls.x_dim5D_distant_cl_low_std, cls.labels_dim5D_distant_cl_low_std = make_blobs(
            n_samples=100, centers=[[0, 0, 0, 0, 0], [10, 10, 10, 10, 10]],
            n_features=5, cluster_std=0.2, random_state=42)
        cls.x_dim5D_distant_cl_high_std, cls.labels_dim5D_distant_cl_high_std = make_blobs(
            n_samples=100, centers=[[0, 0, 0, 0, 0], [10, 10, 10, 10, 10]],
            n_features=5, cluster_std=2.0, random_state=42)


class DunnIndexTest(SetUpBlob):
    def test_cluster_separation(self):
        # 2D
        dunn2D_close1 = calculate_dunn_index(self.x_dim2D_close_cl_low_std, self.labels_dim2D_close_cl_low_std)
        dunn2D_dist1 = calculate_dunn_index(self.x_dim2D_distant_cl_low_std, self.labels_dim2D_distant_cl_low_std)
        self.assertGreater(dunn2D_dist1, dunn2D_close1)

        dunn2D_close2 = calculate_dunn_index(self.x_dim2D_close_cl_high_std, self.labels_dim2D_close_cl_high_std)
        dunn2D_dist2 = calculate_dunn_index(self.x_dim2D_distant_cl_high_std, self.labels_dim2D_distant_cl_high_std)
        self.assertGreater(dunn2D_dist2, dunn2D_close2)

        # 5D
        dunn5D_close1 = calculate_dunn_index(self.x_dim5D_close_cl_low_std, self.labels_dim5D_close_cl_low_std)
        dunn5D_dist1 = calculate_dunn_index(self.x_dim5D_distant_cl_low_std, self.labels_dim5D_distant_cl_low_std)
        self.assertGreater(dunn5D_dist1, dunn5D_close1)

        dunn5D_close2 = calculate_dunn_index(self.x_dim5D_close_cl_high_std, self.labels_dim5D_close_cl_high_std)
        dunn5D_dist2 = calculate_dunn_index(self.x_dim5D_distant_cl_high_std, self.labels_dim5D_distant_cl_high_std)
        self.assertGreater(dunn5D_dist2, dunn5D_close2)

    def test_single_cluster(self):
        # 2D
        for label in np.unique(self.labels_dim2D_close_cl_low_std):
            cluster = self.x_dim2D_close_cl_low_std[self.labels_dim2D_close_cl_low_std == label]
            dunn = calculate_dunn_index(cluster, np.zeros(len(cluster)))
            self.assertAlmostEqual(dunn, 0)

        for label in np.unique(self.labels_dim2D_close_cl_high_std):
            cluster = self.x_dim2D_close_cl_high_std[self.labels_dim2D_close_cl_high_std == label]
            dunn = calculate_dunn_index(cluster, np.zeros(len(cluster)))
            self.assertAlmostEqual(dunn, 0)

        # 5D
        for label in np.unique(self.labels_dim5D_close_cl_low_std):
            cluster = self.x_dim5D_close_cl_low_std[self.labels_dim5D_close_cl_low_std == label]
            dunn = calculate_dunn_index(cluster, np.zeros(len(cluster)))
            self.assertAlmostEqual(dunn, 0)

    def test_diff_cluster_std(self):
        dunn2D_low_std = calculate_dunn_index(self.x_dim2D_close_cl_low_std, self.labels_dim2D_close_cl_low_std)
        dunn2D_high_std = calculate_dunn_index(self.x_dim2D_close_cl_high_std, self.labels_dim2D_close_cl_high_std)
        self.assertGreater(dunn2D_low_std, dunn2D_high_std)

        dunn5D_low_std = calculate_dunn_index(self.x_dim5D_close_cl_low_std, self.labels_dim5D_close_cl_low_std)
        dunn5D_high_std = calculate_dunn_index(self.x_dim5D_close_cl_high_std, self.labels_dim5D_close_cl_high_std)
        self.assertGreater(dunn5D_low_std, dunn5D_high_std)


class SilhouetteIndexTest(SetUpBlob):
    def test_value_range(self):
        silhouette = calculate_silhouette(self.x_dim2D_close_cl_low_std, self.labels_dim2D_close_cl_low_std)
        self.assertGreaterEqual(silhouette, -1)
        self.assertLessEqual(silhouette, 1)

    def test_cluster_separation(self):
        sil_close2D = calculate_silhouette(self.x_dim2D_close_cl_low_std, self.labels_dim2D_close_cl_low_std)
        sil_dist2D = calculate_silhouette(self.x_dim2D_distant_cl_low_std, self.labels_dim2D_distant_cl_low_std)
        self.assertGreater(sil_dist2D, sil_close2D)

        sil_close5D = calculate_silhouette(self.x_dim5D_close_cl_low_std, self.labels_dim5D_close_cl_low_std)
        sil_dist5D = calculate_silhouette(self.x_dim5D_distant_cl_low_std, self.labels_dim5D_distant_cl_low_std)
        self.assertGreater(sil_dist5D, sil_close5D)

    def test_diff_cluster_std(self):
        sil_low_std = calculate_silhouette(self.x_dim2D_close_cl_low_std, self.labels_dim2D_close_cl_low_std)
        sil_high_std = calculate_silhouette(self.x_dim2D_close_cl_high_std, self.labels_dim2D_close_cl_high_std)
        self.assertGreater(sil_low_std, sil_high_std)

    def test_single_cluster(self):
        with self.assertRaises(ValueError):
            calculate_silhouette(self.x_dim2D_close_cl_low_std, np.zeros(len(self.x_dim2D_close_cl_low_std)))


class HarabaszTest(SetUpBlob):
    def test_cluster_separation(self):
        chi2D_close1 = calculate_harabasz(self.x_dim2D_close_cl_low_std, self.labels_dim2D_close_cl_low_std)
        chi2D_dist1 = calculate_harabasz(self.x_dim2D_distant_cl_low_std, self.labels_dim2D_distant_cl_low_std)
        self.assertGreater(chi2D_dist1, chi2D_close1)

        chi2D_close2 = calculate_harabasz(self.x_dim2D_close_cl_high_std, self.labels_dim2D_close_cl_high_std)
        chi2D_dist2 = calculate_harabasz(self.x_dim2D_distant_cl_high_std, self.labels_dim2D_distant_cl_high_std)
        self.assertGreater(chi2D_dist2, chi2D_close2)

        chi5D_close1 = calculate_harabasz(self.x_dim5D_close_cl_low_std, self.labels_dim5D_close_cl_low_std)
        chi5D_dist1 = calculate_harabasz(self.x_dim5D_distant_cl_low_std, self.labels_dim5D_distant_cl_low_std)
        self.assertGreater(chi5D_dist1, chi5D_close1)

        chi5D_close2 = calculate_harabasz(self.x_dim5D_close_cl_high_std, self.labels_dim5D_close_cl_high_std)
        chi5D_dist2 = calculate_harabasz(self.x_dim5D_distant_cl_high_std, self.labels_dim5D_distant_cl_high_std)
        self.assertGreater(chi5D_dist2, chi5D_close2)


if __name__ == '__main__':
    main()