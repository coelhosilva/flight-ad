from tqdm import tqdm
from flight_ad.learn import StatisticalLearner
from flight_ad.wrangling import DataWrangler
from flight_ad.utils.data import DataBinder
from functools import lru_cache

__all__ = ['AnomalyDetectionPipeline', 'bind_and_wrangle']


def bind_and_wrangle(binder, wrangler):
    """Iterate over data in binder and apply wrangler's transformations."""
    ds = []
    for _, d in tqdm(binder.iterdata()):
        ds.append(wrangler.compose(d))
    return ds


class AnomalyDetectionPipeline:
    def __init__(self, binder: DataBinder, wrangler: DataWrangler, learner: StatisticalLearner):
        """Init AnomalyDetectionPipeline with binder, wrangler, and learner."""
        self.binder = binder
        self.wrangler = wrangler
        self.learner = learner

    @lru_cache(maxsize=None)
    def _wrangle(self):
        return bind_and_wrangle(self.binder, self.wrangler)

    def fit(self, y=None, verbose=False):
        self.learner.fit(self._wrangle(), y=y)
        return self

    @lru_cache(maxsize=None)
    def fit_predict(self, y=None, verbose=False):
        return self.learner.fit_predict(self._wrangle(), y=y)

    def fit_transform(self, y=None, verbose=False):
        return self.learner.fit_transform(self._wrangle(), y=y)

    def predict(self, binder):
        return self.learner.predict(bind_and_wrangle(binder, self.wrangler))

    def __str__(self):
        """String version of the class for printing."""
        return f"""
        AnomalyDetectionPipeline:
            binder: {self.binder.__str__()}
            wrangler: {self.wrangler.__str__()}
            learner: {self.learner.__str__()}
        """

    def __repr__(self):
        """String representation of the class."""
        return f"AnomalyDetectionPipeline(binder={self.binder.__repr__()}, wrangler={self.wrangler.__repr__()}, " \
               f"learner={self.learner.__repr__()})"
