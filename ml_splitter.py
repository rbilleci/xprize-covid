import pandas as pd
from constants import DATE
from xlogger import log


def split(df: pd.DataFrame, days_for_validation: int, days_for_test: int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # First, sort the data by date
    df = df.sort_values(DATE)

    # Determine the maximum date
    date_start_test = df[DATE].max() - pd.to_timedelta(days_for_test - 1, unit='d')
    date_start_validation = date_start_test - pd.to_timedelta(days_for_validation, unit='d')

    df_train = df[df[DATE] < date_start_validation]
    df_validation = df[(df[DATE] >= date_start_validation) & (df[DATE] < date_start_test)]
    df_test = df[df[DATE] >= date_start_test]

    # Debug the outpoint
    log(f"Training Range:   {df_train[DATE].min().date()} - {df_train[DATE].max().date()}")
    log(f"Validation Range: {df_validation[DATE].min().date()} - {df_validation[DATE].max().date()}")
    log(f"Test Range:       {df_test[DATE].min().date()} - {df_test[DATE].max().date()}")

    # Sanity Check
    if len(df.index) != len(df_train.index) + len(df_validation.index) + len(df_test.index):
        raise Exception('entries do not add up')

    return df_train, df_validation, df_test


# reserves the end days for test, but gives a random split between train/val
def split_random_with_reserved_test(df: pd.DataFrame,
                                    days_for_test: int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # First, sort the data by date
    df = df.sort_values(DATE)

    # Determine the maximum date
    date_start_test = df[DATE].max() - pd.to_timedelta(days_for_test - 1, unit='d')
    df_test = df[df[DATE] >= date_start_test]
    df_train_and_validate = df[df[DATE] < date_start_test]
    df_train, df_validation = pd.np.split(df_train_and_validate.sample(frac=1), [int(.6 * len(df_train_and_validate))])

    # Sanity Check
    if len(df.index) != len(df_train.index) + len(df_validation.index) + len(df_test.index):
        raise Exception('entries do not add up')

    return df_train, df_validation, df_test


def split_random(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    return pd.np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
