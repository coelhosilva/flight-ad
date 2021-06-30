from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

__all__ = ['clustering_info', 'silhouette']


def clustering_info(clusterer):
    """Retrieve clustering info."""
    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    return labels, n_clusters, n_noise


def silhouette(X, labels):
    """Calculate average and sample silhouette values."""
    try:
        silhouette_avg = silhouette_score(X, labels)
        sample_silhouette_values = silhouette_samples(X, labels)
    except ValueError:
        silhouette_avg = np.nan
        sample_silhouette_values = []

    return silhouette_avg, sample_silhouette_values

