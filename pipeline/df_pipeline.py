from datetime import date, timedelta

import pandas as pd

import oxford_loader
from datasets_constants import PATH_DATA_BASELINE, REFERENCE_COUNTRIES_AND_REGIONS, DATE_LOWER_BOUND
from oxford_constants import COUNTRY_NAME, REGION_NAME, DATE, NPI_COLUMNS, C1, C2, C3, C4, C5, C6, C7, C8, H1, \
    H2, H3, H6, INDEX_COLUMNS, GEO_ID, IS_SPECIALTY, LABEL, PREDICTED_NEW_CASES
from pipeline import df_00_splitter, df_10_data_timeinfo, df_11_countryinfo, df_60_imputer, df_70_label, df_80_scaler, \
    df_90_ohe


def get_datasets_for_training(fn: str,
                              days_for_validation: int,
                              days_for_test: int,
                              column_to_predict: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = oxford_loader.load(fn)
    train, validation, test = df_00_splitter.split(df, days_for_validation, days_for_test)
    train = pipeline_for_training(train, column_to_predict)
    validation = pipeline_for_training(validation, column_to_predict)
    test = pipeline_for_training(test, column_to_predict)
    return train, validation, test


def pipeline_for_training(df: pd.DataFrame, column_to_predict: str) -> pd.DataFrame:
    df = df_10_data_timeinfo.apply(df)
    df = df_11_countryinfo.apply(df)
    df = df_60_imputer.apply(df)
    df = df_70_label.apply(df, column_to_predict)  # apply label before scaling, so that it is not scaled twice
    df = df_80_scaler.apply(df)
    df = df_90_ohe.apply(df)

    # Add the label
    df_label = df[LABEL]
    df = df.drop(labels=[LABEL], axis=1)
    df.insert(0, LABEL, df_label)

    # Drop columns that are not used
    df = df.drop(GEO_ID, axis=1)
    return df


def get_dataset_for_prediction(start_date: date,
                               end_date: date,
                               path_future_data: str) -> pd.DataFrame:
    """ Get the baseline data, determine the max date, and set the initial window to be used """
    df = oxford_loader.load(PATH_DATA_BASELINE, load_for_prediction=True)
    df = df.set_index(INDEX_COLUMNS, drop=True)

    """ Fill in the data frame with missing rows for all past dates and future dates """
    df_geos = pd.read_csv(REFERENCE_COUNTRIES_AND_REGIONS)
    df_geos[REGION_NAME] = df_geos[REGION_NAME].fillna('')
    new_rows = []
    for _, geo in df_geos.iterrows():
        idx_country = geo[COUNTRY_NAME]
        idx_region = geo[REGION_NAME]
        for idx_date in date_range(DATE_LOWER_BOUND, end_date):
            if (idx_country, idx_region, pd.to_datetime(idx_date)) not in df.index:
                new_rows.append({COUNTRY_NAME: idx_country, REGION_NAME: idx_region, DATE: pd.to_datetime(idx_date)})
    df = df.reset_index()
    df = df.append(new_rows, ignore_index=True)

    """ Assign values from the NPI data """
    df_future = oxford_loader.load(path_future_data)
    for _, f in df_future.iterrows():
        idx_c = f[COUNTRY_NAME]
        idx_r = f[REGION_NAME]
        idx_d = f[DATE]
        df.loc[df.index.isin([[idx_c, idx_r, idx_d]]), NPI_COLUMNS] = \
            [f[C1], f[C2], f[C3], f[C4], f[C5], f[C6], f[C7], f[C8], f[H1], f[H2], f[H3], f[H6]]

    """ Fill in missing data """

    """ Add specialty columns """
    df[PREDICTED_NEW_CASES] = 0.0
    df[IS_SPECIALTY] = 0

    return df


def date_range(start_date, end_date):
    days = int((end_date - start_date).days) + 1
    for n in range(int(days)):
        yield start_date + timedelta(n)
