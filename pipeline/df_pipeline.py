import pandas as pd

from loader import oxford_loader, oxford_processor
from oxford_constants import *
from pipeline import df_00_splitter, df_10_data_timeinfo, df_11_countryinfo, df_50_mark_nulls, df_60_imputer, \
    df_70_label, df_80_scaler, df_90_ohe


def process_for_training(fn: str,
                         days_for_validation: int,
                         days_for_test: int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = oxford_loader.load(fn)
    df = oxford_processor.process(df)
    train, validation, test = df_00_splitter.split(df, days_for_validation, days_for_test)
    return process_df(train, True), process_df(validation, True), process_df(test, True)


def process_for_prediction(fn: str) -> pd.DataFrame:
    return process_df(oxford_processor.process(oxford_loader.load(fn)))


def process_df(df, label: bool = False) -> pd.DataFrame:
    # Add additional data fields
    df = df_10_data_timeinfo.apply(df)
    df = df_11_countryinfo.apply(df)

    # Clean existing data
    df = df_50_mark_nulls.apply(df)
    df = df_60_imputer.apply(df)

    # Create the label, before the scaling
    if label:
        df = df_70_label.apply(df)  # apply label before scaling, so that it is not scaled twice

    # Perform scaling and Encoding
    df = df_80_scaler.apply(df)
    df = df_90_ohe.apply(df)

    # As a file step, move label to first column
    if label:
        label = df[LABEL]
        df.drop(labels=[LABEL], axis=1, inplace=True)
        df.insert(0, LABEL, label)
    return df
