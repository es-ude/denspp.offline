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
    def __init__(self, settings: SettingsCluster) -> None:
        """Initialization of module for clustering
        Args:
            settings:   Settings for setting-up the clustering pipeline
        Returns:
            None
        """
        self._settings = settings
        self._cluster = None
        self._n_init_iterations = 10

    def sort_pred2label_data(self, pred_label: np.ndarray, true_label: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
            Sorting
        Args:
            pred_label:     Array with predicted labels
            true_label:     Array with true labels
            features:       Array with features
        Returns:
            Numpy array with sorted output
        """
        label_out = np.zeros(pred_label.shape, dtype=int) - 1

        true_order = np.unique(true_label)
        new_order = np.zeros((self._settings.no_cluster, ), dtype=int) - 1
        for idx, true_id in enumerate(true_order):
            true_pos_id = np.argwhere(true_label == true_id).flatten()

            pred_class = list()
            for i0 in range(10):
                for i1 in true_pos_id:
                    pred_class.append(self._cluster.predict(features[i1, :].reshape((1, -1)))[0])
            pred_class = np.array(pred_class, dtype=int)
            del i0, i1

            # --- Decision
            ids, cnt = np.unique(pred_class, return_counts=True)
            if ids.size == 1:
                new_order[idx] = pred_class[0]
            else:
                new_pos = np.argmax(cnt)
                new_class = ids[new_pos]

                if new_class in new_order:
                    while(new_class in new_order or cnt.size > 1):
                        cnt = np.delete(cnt, new_pos, 0)
                        ids = np.delete(ids, new_pos, 0)
                        new_pos = np.argmax(cnt)
                        new_class = ids[new_pos]

                    new_order[idx] = -1 if cnt.size == 1 else new_class
                else:
                    new_order[idx] = new_class

        # --- Decision: Check for ids with value -1
        set_difference = set(true_order.tolist()) - set(new_order)
        list_difference_result = list(set_difference)
        if len(list_difference_result) == 1:
            pos0 = np.argwhere(new_order == -1).flatten()
            new_order[pos0] = list_difference_result[0]

        # --- Transform
        for idx, id in enumerate(new_order):
            pos = np.argwhere(pred_label == id).flatten()
            label_out[pos] = true_order[idx]

        return label_out

    def cluster_gmm(self, features: np.ndarray) -> np.ndarray:
        """Performing the gaussian mixture model for clusting
        Args:
            features:   Numpy array with features
        Returns:
              Two numpy arrays with infos
        """
        self._cluster = GaussianMixture()
        raise NotImplementedError

    def init_kmeans(self, features: np.ndarray) -> np.ndarray:
        """Initizialiation of kmeans for clustering
        Args:
            features:   Numpy array with features
        Returns:
              Numpy arrays with clusters
        """
        self._cluster = KMeans(
            init="k-means++",
            n_init=self._n_init_iterations if self._n_init_iterations != 0 else 'auto',
            max_iter=100,
            random_state=42,
            n_clusters=self._settings.no_cluster
        ).fit(features)

        return self._cluster.labels_

    def predict_kmeans(self, features: np.ndarray) -> np.ndarray:
        """Predicting classes with kmeans
        Args:
            features:   Numpy array with features
        Returns:
              Numpy arrays with clusters
        """
        if not isinstance(self._cluster, KMeans):
            print("--- Please init KMeans for predicting classes! ---")
            return np.zeros((1,), dtype=int)
        else:
            return self._cluster.predict(features)
