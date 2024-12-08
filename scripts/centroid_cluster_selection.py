import numpy as np
from sklearn.cluster import KMeans

def select_closest_points(features, cluster_labels, centroids, m=1):
    """
    Select the m closest points to each centroid from the clustered features.

    Args:
        features (numpy.ndarray): NxD array of feature embeddings (N samples, D dimensions).
        cluster_labels (numpy.ndarray): Array of cluster labels for each sample.
        centroids (numpy.ndarray): Coordinates of cluster centroids.
        m (int): Number of closest points to select per cluster.

    Returns:
        list: Indices of the selected points.
    """
    selected_indices = []

    # Loop through each cluster
    for cluster_idx, centroid in enumerate(centroids):
        # Get indices of points in the current cluster
        cluster_points_idx = np.where(cluster_labels == cluster_idx)[0]
        cluster_points = features[cluster_points_idx]

        # Compute distances of all points in the cluster to the centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)

        # Find indices of the m closest points to the centroid
        closest_indices = cluster_points_idx[np.argsort(distances)[:m]]
        selected_indices.extend(closest_indices)

    return selected_indices
