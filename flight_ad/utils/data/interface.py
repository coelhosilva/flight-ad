from tqdm import tqdm
from collections import OrderedDict


class DataBinder:
    def __init__(self, bindings=None):
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
        for k, v in self.bindings.items():
            yield k, self.retrieve_data(k)

    def apply_to_all(self, fun):
        data = {}
        for k in tqdm(self.bindings.keys()):
            data[k] = fun(self.retrieve_data(k))

        return data

    def __eq__(self, other):
        return True if self.bindings == other.bindings else False

    def __str__(self):
        return f"""
        Data binder:
        Number of samples: {len(self.bindings.keys())}
        """


"""
old
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict


class DataBinder:
    def __init__(self, data=None):
        if data is None:
            data = OrderedDict()
        self.data = data
        self.data_transformed = None

    def add_data(self, data):
        self.data = {**self.data, **data}

    def retrieve_data(self, key):
        if ('bind_function' not in self.data[key].keys()) or (self.data[key]['bind_function'] is None):
            output = self.data[key]['data']
        else:
            output = self.data[key]['bind_function'](self.data[key]['data'])
        return output

    def retrieve_data_(self, key):
        # Not good.
        d = self.data[key]
        if ('bind_function' not in d.keys()) or (d['bind_function'] is None):
            output = d['data']
        else:
            output = d['bind_function'](d['data'])
        return output

    def retrieve_all_data(self):
        return [self.retrieve_data(k) for k in self.data.keys()]

    def collapse(self):
        return pd.concat(self.retrieve_all_data(), axis=0).reset_index(level=0).rename(columns={'level_0': 'id'}).reset_index(drop=True).copy()

    def iterdata(self):
        for k, v in self.data.items():
            yield k, self.retrieve_data(k)

    def apply_to_all(self, fun, data_category=None):
        if data_category is None:
            for k in tqdm(self.data.keys()):
                self.data[k] = fun(self.retrieve_data(k))
        else:
            for k in tqdm(self.data_transformed.keys()):
                self.data_transformed[k] = fun(self.data_transformed[k])

    def __eq__(self, other):
        return True if self.data == other.data else False

    def __str__(self):
        return f"""
        # Data binder:
        # Number of samples: {len(self.data.keys())}
        # """
