import pandas as pd


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values('date', inplace=True)  # TODO: is this needed, or can we guarantee order?
    grouped = df.groupby(['country_name', 'region_name'])
    grouped = grouped.apply(lambda x: compute_label_for_group('cases', x))
    df = grouped.reset_index(drop=True)
    df['_label'] = df['_label'].apply(lambda x: max(0, -x))  # max 1 million new cases / day, never below 0
    return df


def compute_label_for_group(source_series_name, group):
    group['_label'] = group[source_series_name].diff(-1).fillna(0.0)
    return group
