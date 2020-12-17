import pandas as pd

from oxford_constants import *


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(DATE, inplace=True)
    grouped = df.groupby(GROUP_GEO)
    grouped = grouped.apply(impute_group)
    return grouped.reset_index(drop=True)


def impute_group(group):
    return group.apply(impute_series)


def impute_series(series: pd.Series):
    if series.dtype == 'float64':
        if pd.isnull(series.iloc[0]):
            series.iloc[0] = 0.0  # Set the initial value to zero, if it is undefined
        return series.interpolate(method='linear')  # TODO: HYPER-PARAMETER, to limit area????
    else:
        return series
