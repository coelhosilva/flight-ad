import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import Pipeline
from flight_ad.cluster.opt import calculate_eps
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


__all__ = ['DBSCANPipeline', 'silhouette', 'clustering_info', 'LocalOutlierFactorPipeline']


def retrieve_partial_pipeline(pipeline, up_to):
    if up_to in [step[0] for step in pipeline.steps]:
        last_step = np.array([up_to == step[0] for step in pipeline.steps]).argmax()
        partial_pipeline = pipeline[:last_step+1]
    else:
        raise(Exception("Informed step not available in the pipeline."))
    return partial_pipeline


class DBSCANPipeline:
    def __init__(self, X):
        self.X = X
        self.eps = None
        self.X_transformed = retrieve_partial_pipeline(self._make_pipeline(), 'pca').fit_transform(self.X)
        self.eps, _, _ = calculate_eps(self.X_transformed, mode=np.polyfit, degree=3)

        self.pipeline = self._make_pipeline()

    def fit(self, X=None):
        if X is None:
            X = self.X
        self.pipeline.fit(X)

    def _make_pipeline(self):
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('dbscan', DBSCAN(eps=self.eps, min_samples=1))
            ]
        )


def cluster(X, clustering_algorithm, **kwargs):
    """Perform cluster analysis with DBSCAN."""
    model = clustering_algorithm(**kwargs)

    return model.fit(X)


def clustering_info(db):
    """Grab clustering info."""
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return labels, n_clusters_, n_noise_


def silhouette(X, dbscan):
    try:
        silhouette_avg = silhouette_score(X, dbscan.labels_)
        sample_silhouette_values = silhouette_samples(X, dbscan.labels_)
    except:
        silhouette_avg = np.nan
        sample_silhouette_values = []

    return silhouette_avg, sample_silhouette_values


class LocalOutlierFactorPipeline:
    def __init__(self, X):
        self.labels = None
        self.X = X
        self.X_transformed = retrieve_partial_pipeline(self._make_pipeline(), 'pca').fit_transform(self.X)
        self.pipeline = self._make_pipeline()

    def fit(self, X=None):
        if X is None:
            X = self.X
        # self.pipeline.fit(X)
        self.labels = self.pipeline.fit_predict(X)

    def _make_pipeline(self):
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('model', LocalOutlierFactor())
            ]
        )
