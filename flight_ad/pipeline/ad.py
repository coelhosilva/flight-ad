class AnomalyDetectionPipeline:
    def __init__(self):
        self.statistical_learning_pipeline = None

    def load_data(self, X):
        self.X = X

    def set_labels(self, labels):
        self.labels = labels

    def set_estimator(self, estimator):
        self.estimator = estimator


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
