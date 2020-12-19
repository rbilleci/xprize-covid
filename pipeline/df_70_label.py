import pandas as pd
from oxford_constants import GEO_ID, DATE, PREDICTED_NEW_CASES, CONFIRMED_CASES


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(DATE, inplace=True)
    grouped = df.groupby(GEO_ID).apply(compute_label_for_group)
    df = grouped.reset_index(drop=True)
    df[PREDICTED_NEW_CASES] = df[PREDICTED_NEW_CASES].apply(lambda x: max(0, -x))  # never below 0
    return df


def compute_label_for_group(group):
    group[PREDICTED_NEW_CASES] = group[CONFIRMED_CASES].diff(-1).fillna(0.0)
    return group
