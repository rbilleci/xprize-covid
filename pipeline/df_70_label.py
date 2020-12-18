import pandas as pd
from oxford_constants import GEO_ID, DATE, LABEL


def apply(df: pd.DataFrame, column_to_predict: str) -> pd.DataFrame:
    df.sort_values(DATE, inplace=True)  # TODO: is this needed, or can we guarantee order?
    grouped = df.groupby(GEO_ID)
    grouped = grouped.apply(lambda group: compute_label_for_group(column_to_predict, group))
    df = grouped.reset_index(drop=True)
    df[LABEL] = df[LABEL].apply(lambda x: max(0, -x))  # never below 0
    return df


def compute_label_for_group(column_to_predict, group):
    group[LABEL] = group[column_to_predict].diff(-1).fillna(0.0)
    return group
