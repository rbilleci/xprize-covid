import pandas as pd
from oxford_constants import GEO_ID, DATE, LABEL, CONFIRMED_CASES


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(DATE, inplace=True)  # TODO: is this needed, or can we guarantee order?
    grouped = df.groupby(GEO_ID)
    grouped = grouped.apply(lambda group: compute_label_for_group(group))
    df = grouped.reset_index(drop=True)
    df[LABEL] = df[LABEL].apply(lambda x: max(0, -x))  # never below 0
    return df


def compute_label_for_group(group):
    group[LABEL] = group[CONFIRMED_CASES].diff(-1).fillna(0.0)
    return group
