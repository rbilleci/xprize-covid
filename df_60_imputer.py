import pandas as pd


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values('date', inplace=True)
    grouped = df.groupby(['country_name', 'region_name'])
    grouped = grouped.apply(impute_group)
    return grouped.reset_index(drop=True)


def impute_group(group):
    return group.apply(impute_series)


def impute_series(series: pd.Series):
    if series.dtype == 'float64':
        if pd.isnull(series.iloc[0]):
            series.iloc[0] = 0.0  # Set the initial value to zero, if it is undefined
        return series.interpolate(method='linear')  # TODO: HYPER-PARAMETER
    else:
        return series
