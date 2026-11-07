import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import trustworthiness
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


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
    return float(np.linalg.norm(point1 - point2))


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
    if len(unique_labels) < 2:
        return 0.0

    clusters = [data[labels == label] for label in unique_labels]
    intra_dists = []
    for cluster in clusters:
        if len(cluster) > 1:
            diameter = np.max(cdist(cluster, cluster, metric="euclidean"))
            intra_dists.append(diameter)

    if not intra_dists:
        return 0.0

    # --- Inter-cluster distances ---
    # Min. distance between any two clusters
    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            min_dist = np.min(cdist(clusters[i], clusters[j], metric="euclidean"))
            inter_dists.append(min_dist)

    # Calculate Dunn Index
    max_intra = max(intra_dists)
    min_inter = min(inter_dists) if inter_dists else 0.0

    # Avoid division by zero
    if max_intra == 0:
        return 0.0
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


# ------  Metrics for evaluating structural preservation in the embedding space ----------------------------------------


def calculate_silhouette_preservation_error(
    frames_original_space: np.ndarray, frames_embedding_space: np.ndarray, labels: np.ndarray
) -> float:
    """Calculate the absolute difference between original and embedding silhouette scores.

    Args:
        frames_original_space (np.ndarray): Samples in the original feature space.
        frames_embedding_space (np.ndarray): Samples in the embedding space.
        labels (np.ndarray): Label of each sample.

    Returns:
        float: Absolute difference between the silhouette scores.
    """
    silhouette_embedding_space = calculate_silhouette(frames_embedding_space, labels)
    silhouette_original_space = calculate_silhouette(frames_original_space, labels)

    return abs(silhouette_embedding_space - silhouette_original_space)


def calculate_trustworthiness(
    frames_original_space: np.ndarray, frames_embedding_space: np.ndarray, n_neighbors: int
) -> float:
    """Calculate the trustworthiness of an embedding space.

    Args:
        frames_original_space (np.ndarray): Samples in the original feature space.
        frames_embedding_space (np.ndarray): Samples in the embedding space.
        n_neighbors (int): Number of neighbors considered for each sample.

    Returns:
        float: Trustworthiness score of the embedding space.
    """
    score = trustworthiness(frames_original_space, frames_embedding_space, n_neighbors=n_neighbors)

    return score


def calculate_neighborhood_purity(frames: np.ndarray, labels: np.ndarray, n_neighbors: int = 10) -> float:
    """Calculate the mean neighborhood purity for a given dataset.

    Args:
        frames (np.ndarray): Samples used to determine nearest neighbors.
        labels (np.ndarray): Label of each sample.
        n_neighbors (int): Number of neighbors considered for each sample.

    Returns:
        float: Mean proportion of same-label nearest neighbors.
    """
    model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    neighbor_indices = model.fit(frames).kneighbors(return_distance=False)

    neighbor_indices = neighbor_indices[:, 1:]
    neighbor_labels = labels[neighbor_indices]

    return np.mean(neighbor_labels == labels[:, np.newaxis])
