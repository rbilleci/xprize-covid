import pandas as pd


def split(df: pd.DataFrame,
          days_for_validation: int,
          days_for_test: int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # First, sort the data by date
    df = df.sort_values('date')

    # Determine the maximum date
    date_start_test = df['date'].max() - pd.to_timedelta(days_for_test - 1, unit='d')
    date_start_validation = date_start_test - pd.to_timedelta(days_for_validation, unit='d')

    df_train = df[df.date < date_start_validation]
    df_validation = df[(df.date >= date_start_validation) & (df.date < date_start_test)]
    df_test = df[df.date >= date_start_test]

    # Debug the outpoint
    print(f"Training Range:   {df_train.date.min().date()} - {df_train.date.max().date()}")
    print(f"Validation Range: {df_validation.date.min().date()} - {df_validation.date.max().date()}")
    print(f"Test Range:       {df_test.date.min().date()} - {df_test.date.max().date()}")

    # Sanity Check
    if len(df.index) != len(df_train.index) + len(df_validation.index) + len(df_test.index):
        raise Exception('entries do not add up')

    return df_train, df_validation, df_test
