from sklearn.pipeline import Pipeline
from flight_ad.utils import retrieve_partial_pipeline

__all__ = ['StatisticalLearner']


class StatisticalLearner:
    def __init__(self, steps, memory=None, verbose=False, record=None):
        """Init StatisticalLearner with steps, memory, verbose, and record."""
        if record is None:
            recorded_steps = []
            record = False
        elif isinstance(record, str):
            recorded_steps = [record]
            record = True
        elif isinstance(record, list):
            recorded_steps = record
            record = True
        elif record:
            recorded_steps = [s[0] for s in steps]
        else:
            recorded_steps = []

        self.steps = [*steps['preprocessing'], *steps['training']]
        self.memory = memory
        self.verbose = verbose
        self.record = record
        self.recorded_steps = recorded_steps
        self.pipeline = self._make_pipeline()
        self._init_partial_pipelines()
        self.partial_data = {}

    def _init_partial_pipelines(self):
        self.partial_pipelines = []
        for s in self.recorded_steps:
            self.partial_pipelines.append(
                (s, retrieve_partial_pipeline(self.pipeline, s))
            )

    def _make_pipeline(self):
        return Pipeline(self.steps, memory=self.memory, verbose=self.verbose)

    def fit(self, X, y=None):
        for pipeline in self.partial_pipelines:
            self.partial_data[pipeline[0]] = pipeline[1].fit_transform(X)
        self.pipeline.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        for pipeline in self.partial_pipelines:
            self.partial_data[pipeline[0]] = pipeline[1].fit_transform(X)
        return self.pipeline.fit_transform(X, y)

    def fit_predict(self, X, y=None):
        for pipeline in self.partial_pipelines:
            self.partial_data[pipeline[0]] = pipeline[1].fit_transform(X)
        return self.pipeline.fit_predict(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def __str__(self):
        """String version of the class for printing."""
        return f"""
        StatisticalLearner:
            steps: {self.steps}
            memory: {self.memory}
            verbose: {self.verbose}
            record: {self.record}
            recorded_steps: {self.recorded_steps}
        """

    def __repr__(self):
        """String representation of the class."""
        return f"StatisticalLearner(steps={self.steps}, memory={self.memory}, verbose={self.verbose}, " \
               f"record={self.record}, recorded_steps={self.recorded_steps})"


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    learning_steps = {
        'preprocessing': [
            ('scaler', StandardScaler()),
        ],
        'training': [
            ('pca', PCA())
        ]
    }
    learner = StatisticalLearner(learning_steps, record='scaler')
