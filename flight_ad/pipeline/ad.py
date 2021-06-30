from tqdm import tqdm
from flight_ad.learn import StatisticalLearner
from flight_ad.wrangling import DataWrangler
from flight_ad.utils.data import DataBinder


class AnomalyDetectionPipeline:
    def __init__(self, binder: DataBinder, wrangler: DataWrangler, learner: StatisticalLearner):
        self.binder = binder
        self.wrangler = wrangler
        self.learner = learner

    def run(self, verbose=False):
        ds = []
        for _, d in tqdm(self.binder.iterdata()):
            ds.append(self.wrangler.compose(d))
        self.learner.fit(ds)
        return self
