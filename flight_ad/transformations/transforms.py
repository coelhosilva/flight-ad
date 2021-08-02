import pandas as pd
import numpy as np


__all__ = [
    'insert_missing_data',
    'map_parameters',
    're_reference_column',
    'reshape_df_consecutively',
    'resample_dataframe',
    'reshape_df_interspersed'
]


def insert_missing_data(df_flight, parameter_categories, index):
    """Impute missing flight data."""
    df_flight = df_flight.copy()
    df_continuous = df_flight[index+parameter_categories['continuous']].interpolate().ffill().bfill().set_index(index)
    df_discrete = df_flight[index+parameter_categories['discrete']].ffill().bfill().set_index(index)
    return pd.concat([df_continuous, df_discrete], axis=1).reset_index()


def map_parameters(df, map_dict):
    """
    Map parameters.

    Example:
    map_dict: {'WOW': {'GROUND': 0, 'AIR': 1}, 'FLAP': {'UP':0,'POS 1':1}}
    """
    df = df.copy()
    for k, v in map_dict.items():
        df[k] = df[k].map(v)

    return df


def re_reference_column(df, column, index=None):
    """Re-reference a column by subtracting its value on input index."""
    if index is None:
        index = -1
    df = df.copy()

    return df[column] - df[column].iloc[index]


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
        output[column] = np.interp(resampled_base_column, base_column, df[column])

    return output


def reshape_df_consecutively(df):
    """
    Reads flight data and returns a dataframe containing flight id and the
    flight vector x.
    x = [x_1_t1,x_1_t2,...,x_1_tn,...,x_i_tj,...,x_m_t1,x_m_t2,...,x_m_tn]
    where:
        x_i_tj  <- value of the i-th flight parameter at time tj.
        m       <- total number of parameters
        n       <- number of samples for every parameter
    Similar to Cluster AD Flight.
    """
    df = df.copy()
    return pd.melt(df)['value'].to_numpy()


def reshape_df_interspersed(df):
    """
    Reads flight data and returns a dataframe containing flight id and the
    flight vector x.
    x = [x_1_t1,...,x_m_t1,x_1_t2,...,x_m_t2,...,x_1_tn,...,x_m_tn]
    where:
        x_i_tj  <- value of the i-th flight parameter at time tj.
        m       <- total number of parameters
        n       <- number of samples for every parameter
    Similar to Cluster AD Data Sample.
    """
    df = df.copy()
    return df.unstack().to_frame().sort_index(level=1).values.flatten()
