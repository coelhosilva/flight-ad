from sklearn.base import BaseEstimator, ClassifierMixin


__all__ = ['ClassifierWrapper']


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, criteria, classifying_function=None):
        """Init the ClassifierWrapper with default inputs."""
        if classifying_function is None:
            self.classifying_function = self._binary_classification()
        else:
            self.classifying_function = classifying_function
        self.estimator = estimator
        self.criteria = criteria

    def predict(self, X, *args, **kwargs):
        return self.classifying_function(X, *args, **kwargs)

    def _binary_classification(self, X):
        return (self.estimator.predict(X) >= self.criteria).astype(int)
