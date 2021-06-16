import numpy as np
from sklearn.neighbors import NearestNeighbors
from flight_ad.math.geometry import curvature


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


def calculate_eps_detailed(X, mode=None, **kwargs):
    if mode is None:
        mode = 'empirical'
    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)[:, 1]

    if mode == 'empirical':
        x = np.arange(len(distances))
        y = distances
        point_max = np.argmax(curvature(x, y))
        x_max_curvature = x[point_max]
        y_max_curvature = y[point_max]
    else:
        x0 = np.arange(len(distances))
        y0 = distances
        poly = mode(x0, y0, kwargs['degree'])
        x = np.linspace(0, len(distances), 1000)
        y = np.polyval(poly, x)
        point_max = np.argmax(curvature(x, y))
        x_max_curvature = x[point_max]
        y_max_curvature = y[point_max]
    return y_max_curvature, x, y, x_max_curvature
