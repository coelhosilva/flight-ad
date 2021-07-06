from tqdm import tqdm
from flight_ad.learn import StatisticalLearner
from flight_ad.wrangling import DataWrangler
from flight_ad.utils.data import DataBinder


class AnomalyDetectionPipeline:
    def __init__(self, binder: DataBinder, wrangler: DataWrangler, learner: StatisticalLearner):
        """Init AnomalyDetectionPipeline with binder, wrangler, and learner."""
        self.binder = binder
        self.wrangler = wrangler
        self.learner = learner

    def fit(self, y=None, verbose=False):
        ds = []
        for _, d in tqdm(self.binder.iterdata()):
            ds.append(self.wrangler.compose(d))
        self.learner.fit(ds, y=y)
        return self

    def predict(self, binder):
        ds = []
        for _, d in tqdm(binder.iterdata()):
            ds.append(self.wrangler.compose(d))
        return self.learner.predict(ds)
