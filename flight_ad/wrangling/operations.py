import pandas as pd
import numpy as np

__all__ = [
    'retrieve_all_parameters',
    'insert_missing_data',
    'get_touchdown_index',
    'map_parameters',
    'resample_dataframe',
    'change_column_reference'
]


def retrieve_all_parameters(parameter_info_dict):
    """Retrieve all parameters from parameter dictionary."""
    return sorted({x for v in parameter_info_dict.values() for x in v})


def insert_missing_data(df_flight, parameter_categories, index):
    """Impute missing flight data."""
    df_flight = df_flight.copy()
    df_continuous = df_flight[index + parameter_categories['continuous']].interpolate().ffill().bfill().set_index(index)
    df_discrete = df_flight[index + parameter_categories['discrete']].ffill().bfill().set_index(index)
    return pd.concat([df_continuous, df_discrete], axis=1).reset_index()


def get_touchdown_index(df, pressure_altitude_key, air_ground_key, ground_value):
    """
    Calculate touchdown index.

    :param df:
    :param pressure_altitude_key:
    :param air_ground_key:
    :param ground_value:
    :return:

    Example:
    get_touchdown_index(df, 'ALT', 'WOW', 'GROUND')
    """
    touchdown_index = (df.loc[df[pressure_altitude_key].idxmax():, air_ground_key] == ground_value).idxmax()

    return touchdown_index


def map_parameters(df, map_dict):
    """
    Map parameters.
    map_dict: {'WOW': {'GROUND': 0, 'AIR': 1}, 'FLAP': {'UP':0,'POS 1':1}}
    """
    df = df.copy()
    for k, v in map_dict.items():
        df[k] = df[k].map(v)

    return df


def resample_dataframe(df, samples_per_column, base_column_name=None):
    """
    Resample a dataframe given a desired number of samples.

    :param df:
    :param samples_per_column:
    :param base_column_name:
    :return:
    """
    df = df.copy()

    if base_column_name is None:
        base_column = df.index.values
    else:
        base_column = df[base_column_name]

    output = pd.DataFrame()

    resampled_base_column = np.linspace(np.min(base_column), np.max(base_column), num=samples_per_column)

    for column in df.columns.values:
        if column == base_column_name:
            continue
        try:
            output[column] = np.interp(resampled_base_column, base_column, df[column])
        except:
            output[column] = df[column].iloc[0]

    return output


def change_column_reference(df, column, index=None):
    if index is None:
        index = -1
    df = df.copy()

    return df[column] - df[column].iloc[index]
