from pandas import DataFrame


class FunctionPipeline:
    def __init__(self, steps, memorize=None):
        """Init FunctionPipeline with steps and memorize."""
        if memorize is None:
            memorized_steps = []
            memorize = False
        elif isinstance(memorize, str):
            memorized_steps = [memorize]
            memorize = True
        elif isinstance(memorize, list):
            memorized_steps = memorize
            memorize = True
        elif memorize:
            memorized_steps = [s[0] for s in steps]
        else:
            memorized_steps = []

        self.steps = steps
        self.memorize = memorize
        self.memorized_steps = memorized_steps
        self.results = {s[0]: [] for s in steps}

    def compose(self, x):
        for step in self.steps:
            func = step[1]
            x = func(x)
            if self.memorize and (step[0] in self.memorized_steps):
                self.results[step[0]].append(self._copy_item(x))

        return x

    @property
    def named_steps(self):
        return dict(self.steps)

    @staticmethod
    def _copy_item(x):
        if isinstance(x, DataFrame):
            return x.copy()
        else:
            return x

    def __str__(self):
        """String version of the class for printing."""
        return f"""
        FunctionPipeline:
            steps: {self.steps}
            memorize: {self.memorize}
            results: {self.results if self.memorize else "N/A"}
        """

    def __repr__(self):
        """String representation of the class."""
        return f"FunctionPipeline(steps={self.steps}, memorize={self.memorize})"


class DataWrangler(FunctionPipeline):
    def __init__(self, steps, memorize=None):
        """Init DataWrangler with steps and memorize."""
        super().__init__(steps, memorize)

    def __str__(self):
        """String version of the class for printing."""
        return f"""
        DataWrangler:
            steps: {self.steps}
            memorize: {self.memorize}
            results: {self.results if self.memorize else "N/A"}
        """

    def __repr__(self):
        """String representation of the class."""
        return f"DataWrangler(steps={self.steps}, memorize={self.memorize})"


if __name__ == '__main__':
    import numpy as np

    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    wrangling_steps = [
        ('step1', lambda x: np.max(x, axis=1)),
        ('step2', np.mean),
        ('step3', lambda x: 2 * x)
    ]
    w = DataWrangler(wrangling_steps, memorize=True)
    r = w.compose(data)
