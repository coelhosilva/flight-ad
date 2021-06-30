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
        for f, d in tqdm(self.binder.iterdata()):
            ds.append(self.wrangler.compose(d))
        self.learner.fit(ds)
        return self


"""
class FlightDataLoader:
    def __init__(self):
        pass

class FlightDataWrangler:
    def __init__(self):
        self.steps = []

class AnomalyDetectionModel:
    def __init__(self):
        pass

    def extract_model(self):
        pass
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import DBSCAN


# def data_wrangling():
#     pass


# class DBSCANN(DBSCAN):
#     def __init__(self):
#         super().__init__(eps=1)
"""

# from operator import itemgetter
#
# class FunctionPipeline:
#     def __init__(self, steps):
#         self.steps = steps
#
#     def function_composition(self, origin):
#         destination = origin
#         for func in list(map(itemgetter(1), self.steps)):
#             destination = func(destination)
#         return destination
#
# class DataWrangler(FunctionPipeline):
#     def __init__(self, steps):
#         self.steps = steps