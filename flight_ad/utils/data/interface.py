from tqdm import tqdm
from collections import OrderedDict


class DataBinder:
    def __init__(self, bindings=None):
        """Init DataBinder with bindings."""
        if bindings is None:
            bindings = OrderedDict()
        self.bindings = bindings

    def add_data(self, data):
        self.bindings = {**self.bindings, **data}

    def retrieve_data(self, key):
        if ('bind_function' not in self.bindings[key].keys()) or (self.bindings[key]['bind_function'] is None):
            output = self.bindings[key]['data']
        else:
            output = self.bindings[key]['bind_function'](self.bindings[key]['data'])
        return output

    def retrieve_all_data(self):
        return [self.retrieve_data(k) for k in self.bindings.keys()]

    def iterdata(self):
        for k, _ in self.bindings.items():
            yield k, self.retrieve_data(k)

    def apply_to_all(self, fun):
        data = {}
        for k in tqdm(self.bindings.keys()):
            data[k] = fun(self.retrieve_data(k))

        return data

    def __eq__(self, other):
        """Check equality between two DataBinding objects."""
        return True if self.bindings == other.bindings else False

    def __str__(self):
        """String version of the class for printing."""
        return f"""
        Data binder:
        Number of samples: {len(self.bindings.keys())}
        """

    def __repr__(self):
        """String representation of the class."""
        return f"DataBinder(bindings={self.bindings})"

