import pandas as pd

from oxford_constants import DATE, GEO_ID


def apply(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(DATE).groupby(GEO_ID).apply(impute_missing_values_for_group).reset_index(drop=True)


def impute_missing_values_for_group(group):
    return group.apply(impute_missing_values_for_series)


def impute_missing_values_for_series(series: pd.Series):
    if series.dtype == 'float64':
        if pd.isnull(series.iloc[0]):
            series.iloc[0] = 0.0  # Set the initial value to zero, if it is undefined
        return series.interpolate(method='linear')
    else:
        return series
