from math import sqrt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        point1 (npt.NDArray): Point 1
        point2 (npt.NDArray): Point 2

    Raises:
        ValueError: If point1 and point2 are not of equal size.

    Returns:
        float: Euclidean distance between point1 and point2
    """
    temp_sum = 0
    for f1, f2 in zip(point1, point2, strict=True):
        temp_sum += (f1 - f2) ** 2
    return sqrt(temp_sum)


def calculate_dunn_index(data: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the Dunn-index for a given dataset.

    Args:
        data (np.ndarray): array representing a cluster
                     -> each row describes a sample
                     -> each column represents a different feature
        labels (np.ndarray): label of each sample
    Returns:
        float: Floating with metric value
    """
    unique_labels = np.unique(labels)
    clusters = [data[labels == label] for label in unique_labels]

    intra_dists = [np.max(cdist(cluster, cluster)) for cluster in clusters if len(cluster) > 1]

    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = np.min(cdist(clusters[i], clusters[j]))
            inter_dists.append(dist)

    max_intra = max(intra_dists) if intra_dists else 1e-10
    min_inter = min(inter_dists) if inter_dists else 1e-10
    return float(min_inter / max_intra)


def calculate_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the Silhouette index for a given dataset.

    Args:
        data (np.ndarray): array representing a cluster
                     -> each row describes a sample
                     -> each column represents a different feature
        labels (np.ndarray): label of each sample
    Returns:
        float: Floating with metric value
    """
    return silhouette_score(data, labels)


def calculate_harabasz(data: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the Calinski-Harabasz index for a given dataset.

    Args:
        data (np.ndarray): array representing a cluster
                     -> each row describes a sample
                     -> each column represents a different feature
        labels (np.ndarray): label of each sample
    Returns:
        float: Floating with metric value
    """
    return calinski_harabasz_score(data, labels)
