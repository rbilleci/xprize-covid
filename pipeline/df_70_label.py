import pandas as pd
from oxford_constants import *


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(DATE, inplace=True)  # TODO: is this needed, or can we guarantee order?
    grouped = df.groupby(GROUP_GEO)
    grouped = grouped.apply(lambda x: compute_label_for_group(CASES, x))
    df = grouped.reset_index(drop=True)
    df[LABEL] = df[LABEL].apply(lambda x: max(0, -x))  # max 1 million new cases / day, never below 0
    return df


def compute_label_for_group(source_series_name, group):
    group[LABEL] = group[source_series_name].diff(-1).fillna(0.0)
    return group
