import pandas as pd
import oxford_loader
from constants import *
from xlogger import log
from datetime import date, timedelta


def load_ml_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = oxford_loader.load(PATH_DATA_BASELINE)
    df = prepare_data(df)
    df = truncate_last_day(df)  # for training only, since we don't have the new cases for the final day
    return df


def load_prediction_data(path_future_data: str, end_date: date) -> pd.DataFrame:
    log("loading reference data")
    df = oxford_loader.load(PATH_DATA_BASELINE)

    log("adding rows for future dates")
    df = add_future_rows(df, end_date)

    log("adding data from npi file")
    df = add_future_npi_data(df, path_future_data)

    log("preparing dataset")
    df = prepare_data(df)
    df[IS_SPECIALTY] = 0
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    log("adding null markers")
    df = mark_null_columns(df)
    log("imputing")
    df = impute(df)
    log("computing labels")
    df = compute_label(df)
    return df


def add_future_rows(df: pd.DataFrame, end_date: date) -> pd.DataFrame:
    # Add missing rows from start of time to end-date
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
    # Merge the new rows with the existing data frame, and resort the results by date
    return df.append(pd.DataFrame.from_records(new_rows), ignore_index=True).sort_values(DATE)


def add_future_npi_data(df: pd.DataFrame, path_future_data: str) -> pd.DataFrame:
    # filter out older data from the npi file
    df_future = oxford_loader.load(path_future_data)
    df_future = df_future.loc[(df[DATE] >= pd.to_datetime(DATE_SUBMISSION_CUTOFF))]

    df = df.set_index(INDEX_COLUMNS, drop=False)  # Add the index again :)
    # for idx, f in df_future.iterrows():
    #    idx_c = f[COUNTRY_NAME]
    #    idx_r = f[REGION_NAME]
    #    idx_d = f[DATE]
    #    df.loc[df.index.isin([[idx_c, idx_r, idx_d]]), NPI_COLUMNS] = \
    #        [f[C1], f[C2], f[C3], f[C4], f[C5], f[C6], f[C7], f[C8], f[H1], f[H2], f[H3], f[H6]]
    return df.reset_index(drop=True).sort_values(DATE)  # drop the index again :)


def mark_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    for name in COLUMNS_TO_APPLY_NULL_MARKER:
        df[f"{name}_N"] = df[name].apply(lambda x: (1.0 if pd.isnull(x) else 0.0))
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(GEO_ID).apply(impute_group).reset_index(drop=True)


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
    return df.groupby(GEO_ID).apply(compute_label_for_group).reset_index(drop=True)


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
