import pandas as pd
from oxford_constants import *


def split(df: pd.DataFrame,
          days_for_validation: int,
          days_for_test: int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # First, sort the data by date
    df = df.sort_values(DATE)

    # Determine the maximum date
    date_start_test = df[DATE].max() - pd.to_timedelta(days_for_test - 1, unit='d')
    date_start_validation = date_start_test - pd.to_timedelta(days_for_validation, unit='d')

    df_train = df[df[DATE] < date_start_validation]
    df_validation = df[(df[DATE] >= date_start_validation) & (df[DATE] < date_start_test)]
    df_test = df[df[DATE] >= date_start_test]

    # Debug the outpoint
    print(f"Training Range:   {df_train[DATE].min().date()} - {df_train[DATE].max().date()}")
    print(f"Validation Range: {df_validation[DATE].min().date()} - {df_validation[DATE].max().date()}")
    print(f"Test Range:       {df_test[DATE].min().date()} - {df_test[DATE].max().date()}")

    # Sanity Check
    if len(df.index) != len(df_train.index) + len(df_validation.index) + len(df_test.index):
        raise Exception('entries do not add up')

    return df_train, df_validation, df_test
