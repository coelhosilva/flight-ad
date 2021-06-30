import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from flight_ad.math.geometry import curvature


__all__ = ['DBSCAN']


def calculate_eps(X, mode=None, **kwargs):
    if mode is None:
        mode = 'empirical'
    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)[:, 1]

    if mode == 'empirical':
        x = np.arange(len(distances))
        y = distances
        y_max_curvature = y[np.argmax(curvature(x, y))]
    else:
        x0 = np.arange(len(distances))
        y0 = distances
        poly = mode(x0, y0, kwargs['degree'])
        x = np.linspace(0, len(distances), 1000)
        y = np.polyval(poly, x)
        y_max_curvature = y[np.argmax(curvature(x, y))]
    return y_max_curvature, x, y


class DBSCAN(DBSCAN):
    def __init__(self, eps=None, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def _update_eps(self, X):
        if self.eps is None:
            self.eps, _, _ = calculate_eps(X, np.polyfit, degree=3)

    def fit(self, X, y=None, sample_weight=None):
        self._update_eps(X)
        super().__init__(eps=self.eps, min_samples=self.min_samples, metric='euclidean',
                         metric_params=None, algorithm='auto', leaf_size=30, p=None,
                         n_jobs=None)
        super().fit(X, y, sample_weight)

    def fit_predict(self, X, y=None, sample_weight=None):
        self._update_eps(X)
        super().__init__(eps=self.eps, min_samples=self.min_samples, metric='euclidean',
                         metric_params=None, algorithm='auto', leaf_size=30, p=None,
                         n_jobs=None)
        super().fit_predict(X, y, sample_weight)
