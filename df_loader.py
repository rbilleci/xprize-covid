import pandas as pd

import datasets_additional_info
import oxford_loader
from constants import *
from xlogger import log
from datetime import date, timedelta


def load_ml_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = oxford_loader.load(PATH_DATA_BASELINE)
    df = filter_out_early_data(df)
    df = prepare_data(df)
    df = truncate_last_day(df)  # for training only, since we don't have the new cases for the final day
    return df


def load_prediction_data(path_future_data: str, end_date: date) -> pd.DataFrame:
    df = oxford_loader.load(PATH_DATA_BASELINE)
    df = add_npis(df, path_future_data)
    df = add_missing_dates(df, end_date)
    df = prepare_data(df)
    df[IS_SPECIALTY] = 0
    return df


def filter_out_early_data(df: pd.DataFrame):
    return df[(df[DATE] >= pd.to_datetime(TRAINING_DATA_START_DATE))]


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


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    log("preparing data")
    df = add_population_info(df)
    df = df.sort_values(DATE).groupby(GEO_ID).apply(prepare_data_for_group).reset_index(drop=True)
    df[CONFIRMED_CASES_PER_100K] = df[CONFIRMED_CASES] / df[POPULATION]
    return df


def prepare_data_for_group(group):
    log(f"processing group {group.name}")
    group = group.apply(impute)
    group = label(group)
    group = add_ma(group)
    return group


def impute(series: pd.Series) -> pd.Series:
    if series.dtype == 'float64':
        if pd.isnull(series.iloc[0]):  # set the initial row value to 0, if it is null
            series.iloc[0] = 0.0
        return series.ffill()
    else:
        return series


def label(group):
    group[PREDICTED_NEW_CASES] = group[CONFIRMED_CASES].diff(-1).fillna(0.0).apply(lambda x: max(0, -x))
    return group


def add_ma(group) -> pd.Series:
    add_ma_for_series(group, group[C1], C1)
    add_ma_for_series(group, group[C2], C2)
    add_ma_for_series(group, group[C3], C3)
    add_ma_for_series(group, group[C4], C4)
    add_ma_for_series(group, group[C5], C5)
    add_ma_for_series(group, group[C6], C6)
    add_ma_for_series(group, group[C7], C7)
    add_ma_for_series(group, group[C8], C8)
    add_ma_for_series(group, group[H1], H1)
    add_ma_for_series(group, group[H2], H2)
    add_ma_for_series(group, group[H3], H3)
    add_ma_for_series(group, group[H6], H6)
    add_ma_for_series(group, group[CONFIRMED_CASES], CONFIRMED_CASES)
    return group


def add_ma_for_series(group, series: pd.Series, name) -> None:
    shifted_series = series.shift(COVID_INCUBATION_PERIOD)
    group[name + SUFFIX_MA_DIFF] = shifted_series.diff().ewm(span=MOVING_AVERAGE_SPAN).mean()
    group[name + SUFFIX_MA_DIFF] = group[name + SUFFIX_MA_DIFF].fillna(0.0)


def truncate_last_day(df: pd.DataFrame) -> pd.DataFrame:
    date_max = df[DATE].max() - timedelta(days=DAYS_TO_STRIP_FROM_DATASET)
    return df.loc[(df[DATE] <= date_max)]


def date_range(start_date, end_date):
    days = int((end_date - start_date).days) + 1
    for n in range(int(days)):
        yield start_date + timedelta(n)


def add_population_info(df: pd.DataFrame) -> pd.DataFrame:
    log("adding population info")
    df[POPULATION] = df[GEO_ID].apply(
        lambda x: datasets_additional_info.ADDITIONAL_DATA_GEO[x][POPULATION])
    df[POPULATION_DENSITY] = df[GEO_ID].apply(
        lambda x: datasets_additional_info.ADDITIONAL_DATA_GEO[x][POPULATION_DENSITY])
    return df
