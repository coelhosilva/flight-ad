from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

__all__ = ['FunctionEstimator']


class FunctionEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, fun):
        """Init FunctionEstimator with function fun."""
        self.fun = fun

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(list(map(self.fun, [d for d in X])))
