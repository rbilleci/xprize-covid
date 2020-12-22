import pandas as pd

from datasets_additional_info import *
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
    df = df.sort_values(DATE)

    # Add population info
    df[POPULATION] = df[GEO_ID].apply(lambda geo_id: ADDITIONAL_DATA_GEO[geo_id][POPULATION])
    df[POPULATION_DENSITY] = df[GEO_ID].apply(lambda geo_id: ADDITIONAL_DATA_GEO[geo_id][POPULATION_DENSITY])
    df = df.groupby(GEO_ID).apply(group_impute).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(group_label).reset_index(drop=True)

    # Add Moving Averages for confirmed cases
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, CONFIRMED_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, CONFIRMED_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, CONFIRMED_CASES)).reset_index(drop=True)

    # Add Moving Averages for predicted cases
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, PREDICTED_NEW_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, PREDICTED_NEW_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, PREDICTED_NEW_CASES)).reset_index(drop=True)

    # Add npi column moving averages
    for column in NPI_COLUMNS:
        df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, column)).reset_index(drop=True)
        df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, column)).reset_index(drop=True)
        df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, column)).reset_index(drop=True)
    # drop redundant input
    df = df.drop([C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6], axis=1)
    return df


def group_impute(grouped):
    for series_name in grouped:
        series = grouped[series_name]
        if series.dtype == 'float64':
            if pd.isnull(series.iloc[0]):  # set the initial row value to 0, if it is null
                series.iloc[0] = 0.0
            grouped[series.name] = series.ffill()
    return grouped


def group_label(grouped):
    grouped[PREDICTED_NEW_CASES] = grouped[CONFIRMED_CASES].copy()
    grouped[PREDICTED_NEW_CASES] = grouped[PREDICTED_NEW_CASES].diff(-1).fillna(0.0).apply(lambda x: max(0, -x))
    if USE_CASES_PER_100K:
        population = ADDITIONAL_DATA_GEO[grouped.name][POPULATION]
        grouped[PREDICTED_NEW_CASES] = grouped[PREDICTED_NEW_CASES].apply(lambda value: (1e5 * value / population))
    return grouped


def group_add_ma_a(grouped, name: str):
    return group_add_ma_n(grouped, name, SUFFIX_MA_A, MA_WINDOW_A)


def group_add_ma_b(grouped, name: str):
    return group_add_ma_n(grouped, name, SUFFIX_MA_B, MA_WINDOW_B)


def group_add_ma_c(grouped, name: str):
    return group_add_ma_n(grouped, name, SUFFIX_MA_C, MA_WINDOW_C)


def group_add_ma_n(grouped, name: str, suffix: str, window: int):
    name_ma = name + suffix
    # shift by 1 so we look only at past days
    # NOTE: the shift is also important so we don't include today's predicted data in the value
    grouped[name_ma] = grouped[name].copy().shift(1).bfill().ffill()  # NOTE copy is needed?
    grouped[name_ma] = grouped[name_ma].rolling(window=window, min_periods=1).mean().bfill().ffill()
    return grouped


def truncate_last_day(df: pd.DataFrame) -> pd.DataFrame:
    date_max = df[DATE].max() - timedelta(days=DAYS_TO_STRIP_FROM_DATASET)
    return df.loc[(df[DATE] <= date_max)]


def date_range(start_date, end_date):
    days = int((end_date - start_date).days) + 1
    for n in range(int(days)):
        yield start_date + timedelta(n)
