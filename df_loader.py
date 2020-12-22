import pandas as pd

import oxford_loader
from datasets_geo import *
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
    log("adding population information")
    df[POPULATION] = df[GEO_ID].apply(lambda geo_id: DATA_POPULATION[geo_id][POPULATION])
    df[POPULATION_DENSITY] = df[GEO_ID].apply(lambda geo_id: DATA_POPULATION[geo_id][POPULATION_DENSITY])

    # Add the age info
    log("adding age information")
    if ADD_AGE_BINS:
        df[AGE_R1] = df[GEO_ID].apply(lambda geo_id: DATA_AGE[geo_id][AGE_R1])
        df[AGE_R2] = df[GEO_ID].apply(lambda geo_id: DATA_AGE[geo_id][AGE_R2])
        df[AGE_R3] = df[GEO_ID].apply(lambda geo_id: DATA_AGE[geo_id][AGE_R3])
        df[AGE_R4] = df[GEO_ID].apply(lambda geo_id: DATA_AGE[geo_id][AGE_R4])
        df[AGE_R5] = df[GEO_ID].apply(lambda geo_id: DATA_AGE[geo_id][AGE_R5])

    # Fill missing values
    log("filling missing values")
    df = df.groupby(GEO_ID).apply(group_impute).reset_index(drop=True)

    # Set confirmed cases as percent of population
    if CALCULATE_AS_PERCENT_OF_POPULATION:
        df[CONFIRMED_CASES] = df[CONFIRMED_CASES] / df[POPULATION]

    # Calculate the predicted cases
    log("calculating predicted cases")
    df = df.groupby(GEO_ID).apply(group_label).reset_index(drop=True)

    # Add Temperature and humidity information
    log("adding temperature and humidity information")
    if ADD_SPECIFIC_HUMIDITY_INFO or ADD_TEMPERATURE_INFO:
        df_geo_temps = load_geo_temps()
        df = pd.merge(df, df_geo_temps, how='left', left_on=[GEO_ID, DATE], right_on=[GEO_ID, DATE])
        if ADD_TEMPERATURE_INFO:
            df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, TEMPERATURE)).reset_index(drop=True)
            df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, TEMPERATURE)).reset_index(drop=True)
            df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, TEMPERATURE)).reset_index(drop=True)
        if ADD_SPECIFIC_HUMIDITY_INFO:
            df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, SPECIFIC_HUMIDITY)).reset_index(drop=True)
            df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, SPECIFIC_HUMIDITY)).reset_index(drop=True)
            df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, SPECIFIC_HUMIDITY)).reset_index(drop=True)
        df = df.drop([TEMPERATURE, SPECIFIC_HUMIDITY], axis=1)

    # Add Moving Averages for confirmed cases
    log("adding moving averages for confirmed cases")
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, CONFIRMED_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, CONFIRMED_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, CONFIRMED_CASES)).reset_index(drop=True)

    # Add Moving Averages for predicted cases
    log("adding moving averages for predicted cases")
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, PREDICTED_NEW_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, PREDICTED_NEW_CASES)).reset_index(drop=True)
    df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, PREDICTED_NEW_CASES)).reset_index(drop=True)

    # Add npi column moving averages
    log("adding moving averages for NPI columns")
    for column in NPI_COLUMNS:
        df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_a(group, column)).reset_index(drop=True)
        df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_b(group, column)).reset_index(drop=True)
        df = df.groupby(GEO_ID).apply(lambda group: group_add_ma_c(group, column)).reset_index(drop=True)
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


def load_geo_temps():
    try:
        tmp = pd.read_csv(REFERENCE_TEMPERATURES)
        tmp[REGION_NAME] = tmp[REGION_NAME].fillna('')
        tmp[GEO_ID] = tmp[COUNTRY_NAME] + tmp[REGION_NAME]
        tmp = tmp.drop([COUNTRY_NAME, REGION_NAME], axis=1)
        tmp[DATE] = pd.to_datetime(tmp[DATE])
        return tmp
    except FileNotFoundError:
        tmp = pd.read_csv('work/' + REFERENCE_TEMPERATURES)
        tmp[REGION_NAME] = tmp[REGION_NAME].fillna('')
        tmp[GEO_ID] = tmp[COUNTRY_NAME] + tmp[REGION_NAME]
        tmp = tmp.drop([COUNTRY_NAME, REGION_NAME], axis=1)
        tmp[DATE] = pd.to_datetime(tmp[DATE])
        return tmp
