from flight_ad.wrangling import \
    retrieve_all_parameters,\
    get_touchdown_index,\
    map_parameters,\
    change_column_reference,\
    insert_missing_data,\
    resample_dataframe


def preprocess_flight(df_flight, parameter_categories):
    """Preprocess flight. Filters a portion of it. Maps categorical parameters to numeric values."""
    # Param maps
    param_map = {'WOW': {'GROUND': 0, 'AIR': 1}}

    # Filter a portion of the flight
    df = df_flight[retrieve_all_parameters(parameter_categories)]
    df_filled = insert_missing_data(df, parameter_categories, index=parameter_categories['index'])
    touchdown_index = get_touchdown_index(df_filled, 'ALT', 'WOW', 'GROUND')

    output = df_filled.loc[touchdown_index-300:touchdown_index, ].copy()
    output = output.loc[output['RALT'] < 50, ].loc[:touchdown_index, ]

    # Map parameters
    output = map_parameters(output, param_map)

    return output


def change_col(df):
    col = 'time'
    df[col] = change_column_reference(df, col)
    return df.copy()


def resample(df):
    max_no_samples = 282
    return resample_dataframe(df, samples_per_column=max_no_samples)


def preprocess(df):
    parameter_categories = {
        'continuous': ['RALT', 'CAS', 'ALT'],
        'discrete': ['WOW', 'flight_id'],
        'index': ['time']
    }

    return preprocess_flight(df, parameter_categories)


def select(df):
    cols = ['RALT']

    return df[cols]
