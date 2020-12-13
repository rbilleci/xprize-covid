import numpy as np
import datetime
import math
from pandas import DataFrame as df


def split(df, training_percent, validation_percent, test_percent):
    if not math.isclose(1.0, training_percent + validation_percent + test_percent):
        raise Exception(f"invalid split percentages: {training_percent},  {validation_percent}, {test_percent}")

    # First, sort the data by date
    df = df.sort_values('date')

    # Determine the minimum and maximum dates in the dataset
    days = df.date.max().toordinal() - df.date.min().toordinal()
    date_start_train = df.date.min().toordinal()
    date_start_validation = date_start_train + int(days * training_percent)
    date_start_test = date_start_validation + int(days * validation_percent)

    # Add a temporary day column to do the split
    df['day'] = df.date.apply(lambda x: x.toordinal())

    # Training Split
    df_train = df[df.day < date_start_validation]
    df_train = df_train.drop('day', axis=1)

    # Validation Split
    df_validation = df[(df.day >= date_start_validation) & (df.day < date_start_test)]
    df_validation = df_validation.drop('day', axis=1)

    # Test Split
    df_test = df[df.day >= date_start_test]
    df_test = df_test.drop('day', axis=1)

    # Cleanup
    df = df.drop('day', axis=1)

    # Debug the outpoint
    print(f"Training Range:   {df_train.date.min().date()} - {df_train.date.max().date()}")
    print(f"Validation Range: {df_validation.date.min().date()} - {df_validation.date.max().date()}")
    print(f"Test Range:       {df_test.date.min().date()} - {df_test.date.max().date()}")

    # Sanity Check
    if len(df.index) != len(df_train.index) + len(df_validation.index) + len(df_test.index):
        raise Exception('entries do not add up')

    return df_train, df_validation, df_test
