import dataclasses
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


@dataclasses.dataclass
class SettingsCluster:
    """"Individuall data class to configure clustering"""
    no_cluster: int


RecommendedSettingsCluster = SettingsCluster(
    no_cluster=3
)


class Clustering:
    def __init__(self, settings: SettingsCluster):
        self.settings = settings

    def __dist(self, x, y):
        return sum((xi - yi) ** 2 for xi, yi in zip(x, y))

    def cluster_gmm(self, features: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performing the gaussian mixture model for clusting"""
        raise NotImplementedError

    def cluster_kmeans(self, features: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performing kmeans for clustering"""
        cluster = KMeans(
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
            n_clusters=self.settings.no_cluster
        ).fit(features)

        return cluster.labels_