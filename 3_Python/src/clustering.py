import dataclasses
import numpy as np
from sklearn.cluster import KMeans

@dataclasses.dataclass
class SettingsCluster:
    """"Individuall data class to configure clustering"""
    no_cluster: int

@dataclasses.dataclass
class RecommendedSettingsCluster(SettingsCluster):
    """Recommended values to configure clustering with standard values"""
    def __init__(self):
        super().__init__(
            no_cluster=3
        )

class Clustering:
    def __init__(self, settings: SettingsCluster):
        self.settings = settings

    def cluster_kmeans(self, features: np.ndarray):
        """Performing kmeans as clustering"""
        sse = []

        cluster = KMeans(
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
            n_clusters=self.settings.no_cluster
        )
        cluster.fit(features)
        sse.append(cluster.inertia_)

        results = cluster.labels_
        number = cluster.n_clusters
        return (results, number, sse)

    def calc_spiketicks(self, uin: np.ndarray, xpos: np.ndarray, cluster_id: np.ndarray) -> np.ndarray:
        """Determining spike ticks with cluster results"""
        cluster_no = np.unique(cluster_id)
        ticks = np.zeros(shape=(cluster_no.size, uin.size), dtype=int)

        idx = 0
        for val in xpos:
            ticks[cluster_id[idx], val] = 1
            idx += 1
        return ticks