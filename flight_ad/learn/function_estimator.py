from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

__all__ = ['FunctionTransformer']


class FunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fun, map_iter=True):
        """Init FunctionEstimator with function fun."""
        self.fun = fun
        self.map_iter = map_iter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.map_iter:
            return np.array(list(map(self.fun, [d for d in X])))
        else:
            return self.fun(X)
