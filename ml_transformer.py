import math
from calendar import monthrange

import numpy as np
import pandas as pd

import datasets
import datasets_working_days
from constants import *


def transform(df: pd.DataFrame, for_prediction=True) -> pd.DataFrame:
    df = transform_date(df)
    df = transform_country(df)
    df = transform_value_scale(df)
    df = transform_encoding(df)
    df = transform_column_order(df)
    df = filter_input_columns(df, for_prediction)
    return df


def transform_date(df: pd.DataFrame) -> pd.DataFrame:
    df[DAY_OF_WEEK] = df[DATE].apply(datetime.date.weekday)
    df[DAY_OF_WEEK_SIN] = df[DATE].apply(lambda x: math.sin(2 * np.pi * x.weekday() / 6))
    df[DAY_OF_WEEK_COS] = df[DATE].apply(lambda x: math.cos(2 * np.pi * x.weekday() / 6))
    df[DAY_OF_YEAR_SIN] = df[DATE].apply(lambda x: math.sin(2 * np.pi * x.timetuple().tm_yday / 365))
    df[DAY_OF_YEAR_COS] = df[DATE].apply(lambda x: math.cos(2 * np.pi * x.timetuple().tm_yday / 365))
    df[DAY_OF_MONTH_SIN] = df[DATE].apply(lambda x: math.sin(2 * np.pi * x.day / monthrange(x.year, x.month)[1]))
    df[DAY_OF_MONTH_COS] = df[DATE].apply(lambda x: math.cos(2 * np.pi * x.day / monthrange(x.year, x.month)[1]))
    return df


def transform_country(df: pd.DataFrame) -> pd.DataFrame:
    df[CONTINENT] = df[COUNTRY_NAME].apply(lambda x: datasets.country_name_to_continent[x])
    df[IS_WORKING_DAY_TODAY] = df[[COUNTRY_NAME, DATE]].apply(is_working_day_today, axis=1)
    df[IS_WORKING_DAY_TOMORROW] = df[[COUNTRY_NAME, DATE]].apply(is_working_day_tomorrow, axis=1)
    df[IS_WORKING_DAY_YESTERDAY] = df[[COUNTRY_NAME, DATE]].apply(is_working_day_yesterday, axis=1)
    return df


def transform_value_scale(df: pd.DataFrame) -> pd.DataFrame:
    # Scale Standard Numeric Values
    for e in df.items():
        name, series = str(e[0]), e[1]

        # Standard NPI data
        if name in INPUT_SCALE.keys():
            if CALCULATE_PER_100K and name == PREDICTED_NEW_CASES:
                df[name] = df[name].apply(lambda x: scale_value(x, 0.0, 1e5))
            elif name.endswith(SUFFIX_MA_DIFF):
                df[name] = df[name].apply(lambda x: scale_value(x, -INPUT_SCALE.get(name), INPUT_SCALE.get(name)))
            else:
                df[name] = df[name].apply(lambda x: scale_value(x, 0.0, INPUT_SCALE.get(name)))

        # For standard sin/cos
        elif name.endswith('_sin') or name.endswith('_cos'):
            df[name] = df[name].apply(lambda x: scale_value(x, -1.0, 1.0))

    # Special handling for dates
    df[DATE] = df[DATE].apply(lambda x: scale_value(x.toordinal(), DATE_ORDINAL_LOWER_BOUND, DATE_ORDINAL_UPPER_BOUND))

    return df


def transform_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = ohe(df, COUNTRY_NAME,
             datasets.ALLOWED_COUNTRIES)  # TODO: after hash encoding, we can remove the allowed countries variable
    df = ohe(df, REGION_NAME, datasets.region_name_to_region_code.keys())
    df = ohe(df, CONTINENT, datasets.country_name_to_continent.values())
    df = ohe(df, DAY_OF_WEEK, range(0, 7))
    return df


# TODO: factory out, or may multiple encoding functions
def ohe(df: pd.DataFrame, column: str, values) -> pd.DataFrame:
    df = df.join(pd.get_dummies(df[column], prefix=column, dtype=np.float64))
    for value in values:
        class_name = f"{column}_{value}"
        if class_name not in df.keys():
            df[class_name] = 0.0
    df = df.drop(column, axis=1)
    return df


def transform_column_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(sorted(df.columns), axis=1)  # Sort columns by name
    df_label = df[PREDICTED_NEW_CASES]
    df = df.drop(labels=[PREDICTED_NEW_CASES], axis=1)
    df.insert(0, PREDICTED_NEW_CASES, df_label)
    return df


def filter_input_columns(df: pd.DataFrame, for_prediction: bool) -> pd.DataFrame:
    to_drop = [GEO_ID, IS_SPECIALTY, PREDICTED_NEW_CASES] if for_prediction else [GEO_ID, IS_SPECIALTY]
    return df.drop(to_drop, errors='ignore', axis=1)


def scale_value(x, min_value, max_value):
    x = (x - min_value) / (max_value - min_value)
    x = max(0.0, x)
    x = min(1.0, x)
    return x


def is_working_day_today(x):
    return is_working_day(x[COUNTRY_NAME], x[DATE].toordinal())


def is_working_day_tomorrow(x):
    return is_working_day(x[COUNTRY_NAME], x[DATE].toordinal() + 1)


def is_working_day_yesterday(x):
    return is_working_day(x[COUNTRY_NAME], x[DATE].toordinal() - 1)


def is_working_day(country_name: str, date_as_ordinal: int):
    if country_name in datasets_working_days.non_working_days:
        country_data = datasets_working_days.non_working_days[country_name]
        if date_as_ordinal not in country_data.keys():
            return 1.0
    return 0.0
