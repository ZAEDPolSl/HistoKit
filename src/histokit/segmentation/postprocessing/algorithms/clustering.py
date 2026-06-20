import numpy as np

def cluster_regions(data, max_iters=100, tol=1e-4):
    """
    Cluster regions areas using a single-threaded KMeans algorithm with deterministic initialization.

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        1D data vector to be clustered.
    max_iters : int, optional
        Maximum number of iterations (default is 100).
    tol : float, optional
        Tolerance for convergence (default is 1e-4).

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels assigned to each sample.
    centers : ndarray, shape (n_clusters,)
        Coordinates of cluster centers.
    """

    k=2
    data = np.array(data, dtype=float)
    centroids = np.linspace(data.min(), data.max(), k)

    if len(data) < k or np.all(data == data[0]):
        return np.zeros(len(data), dtype=int), np.array([data.mean()])

    for _ in range(max_iters):

        distances = np.abs(data[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)

        # Default matlab implementation - when cluster is empty, create a new cluster center by assigning
        # its centroid position to the furthest point of another clusters
        if len(set(labels)) < 2:
            empty_label = set(range(k)) - set(labels)
            idx_non_empty = np.argmax(distances[labels != empty_label])
            labels[idx_non_empty] = empty_label


        # Calculate a new centroid position by calculating the mean
        # of samples assigned to this cluster.
        new_centroids = np.array([
            data[labels == i].mean() for i in range(k)
        ])

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return labels, centroids