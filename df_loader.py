import pandas as pd
import oxford_loader
from constants import *
from xlogger import log
from datetime import date, timedelta


def load_ml_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = oxford_loader.load(PATH_DATA_BASELINE)
    df = clean_data(df)
    df = add_moving_averages(df)
    df = truncate_last_day(df)  # for training only, since we don't have the new cases for the final day
    return df.sort_values(DATE)


def load_prediction_data(path_future_data: str, end_date: date) -> pd.DataFrame:
    df = oxford_loader.load(PATH_DATA_BASELINE)
    df = add_npis(df, path_future_data)
    df = add_missing_dates(df, end_date)
    df = clean_data(df)
    df = add_moving_averages(df)
    df[IS_SPECIALTY] = 0
    return df.sort_values(DATE)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = mark_null_columns(df)
    df = impute(df)
    df = compute_label(df)
    return df


def add_npis(df: pd.DataFrame, path_future_data: str) -> pd.DataFrame:
    # Load the NPI file and remove any entries AFTER the MAX date in our reference data
    log("adding NPIs")
    df_future_start_date = df[DATE].max()
    df_future = oxford_loader.load(path_future_data)
    df_future = df_future.loc[(df_future[DATE] > df_future_start_date)]
    df = df.append(df_future, ignore_index=True)
    return df


def add_missing_dates(df: pd.DataFrame, end_date: date) -> pd.DataFrame:
    log(f"adding missing rows")
    df = df.set_index(INDEX_COLUMNS, drop=False)
    new_rows = []
    for _, geo in oxford_loader.df_geos.iterrows():
        idx_country = geo[COUNTRY_NAME]
        idx_region = geo[REGION_NAME]
        for idx_date in date_range(DATE_LOWER_BOUND, end_date):
            if (idx_country, idx_region, pd.to_datetime(idx_date)) not in df.index:
                new_rows.append({COUNTRY_NAME: idx_country,
                                 REGION_NAME: idx_region,
                                 DATE: pd.to_datetime(idx_date),
                                 GEO_ID: geo[GEO_ID],
                                 PREDICTED_NEW_CASES: 0.0,
                                 CONFIRMED_CASES: 0.0})
    df = df.reset_index(drop=True)
    # Merge the new rows with the existing data frame
    log(f"filling in dataset with {len(new_rows)} rows")
    return df.append(pd.DataFrame.from_records(new_rows), ignore_index=True)


def mark_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("marking null columns")
    for name in COLUMNS_TO_APPLY_NULL_MARKER:
        df[f"{name}_N"] = df[name].apply(lambda x: (1.0 if pd.isnull(x) else 0.0))
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    log("imputing")
    return df.sort_values(DATE).groupby(GEO_ID).apply(impute_group).reset_index(drop=True)


def impute_group(group):
    return group.apply(impute_group_series)


def impute_group_series(series: pd.Series):
    if series.dtype == 'float64':
        if pd.isnull(series.iloc[0]):  # set the initial row value to 0, if it is null
            series.iloc[0] = 0.0
        return series.ffill()
    else:
        return series


def compute_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(DATE).groupby(GEO_ID).apply(compute_label_for_group).reset_index(drop=True)


def compute_label_for_group(group: pd.DataFrame) -> pd.DataFrame:
    group[PREDICTED_NEW_CASES] = group[CONFIRMED_CASES].diff(-1).fillna(0.0).apply(lambda x: max(0, -x))
    return group


def truncate_last_day(df: pd.DataFrame) -> pd.DataFrame:
    date_max = df[DATE].max() - timedelta(days=DAYS_TO_STRIP_FROM_DATASET)
    return df.loc[(df[DATE] <= date_max)]


def date_range(start_date, end_date):
    days = int((end_date - start_date).days) + 1
    for n in range(int(days)):
        yield start_date + timedelta(n)


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    log("adding ewma")
    return df.sort_values(DATE).groupby(GEO_ID).apply(add_moving_averages_group).reset_index(drop=True)


def add_moving_averages_group(group):
    return group.apply(lambda series: add_moving_averages_group_series(group, series))


def add_moving_averages_group_series(group: pd.DataFrame, series: pd.Series) -> pd.Series:
    add_moving_average_group_series_ma(group, group[C1], C1)
    add_moving_average_group_series_ma(group, group[C2], C2)
    add_moving_average_group_series_ma(group, group[C3], C3)
    add_moving_average_group_series_ma(group, group[C4], C4)
    add_moving_average_group_series_ma(group, group[C5], C5)
    add_moving_average_group_series_ma(group, group[C6], C6)
    add_moving_average_group_series_ma(group, group[C7], C7)
    add_moving_average_group_series_ma(group, group[C8], C8)
    add_moving_average_group_series_ma(group, group[H1], H1)
    add_moving_average_group_series_ma(group, group[H2], H2)
    add_moving_average_group_series_ma(group, group[H3], H3)
    add_moving_average_group_series_ma(group, group[H6], H6)
    add_moving_average_group_series_ma(group, group[CONFIRMED_CASES], CONFIRMED_CASES)
    add_moving_average_group_series_ma(group, group[PREDICTED_NEW_CASES], PREDICTED_NEW_CASES)
    return series


def add_moving_average_group_series_ma(group: pd.DataFrame, series: pd.Series, name) -> None:
    shifted_series = series.shift(COVID_INCUBATION_PERIOD)
    # calculate the ewm
    group[name + SUFFIX_MA] = shifted_series.ewm(span=MOVING_AVERAGE_SPAN).mean()
    group[name + SUFFIX_MA] = group[name + SUFFIX_MA].fillna(0.0)
    # take a diff on the moving average
    # lets not track the diff for now, unless it is confirmed or predicted cases
    if name in [PREDICTED_NEW_CASES, CONFIRMED_CASES]:
        group[name + SUFFIX_MA_DIFF] = group[name + SUFFIX_MA].diff()
        group[name + SUFFIX_MA_DIFF] = group[name + SUFFIX_MA_DIFF].fillna(0.0)
