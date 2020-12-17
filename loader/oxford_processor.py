import pandas as pd
from datasets import datasets


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    for name in df.columns:
        if name not in datasets.allowed_columns_on_import:
            df.drop(name, axis=1, inplace=True)
    return df


def filter_countries(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['country_name'].isin(datasets.allowed_countries)]
    return df


def map_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={
        "C1_School closing": "c1",
        "C2_Workplace closing": "c2",
        "C3_Cancel public events": "c3",
        "C4_Restrictions on gatherings": "c4",
        "C5_Close public transport": "c5",
        "C6_Stay at home requirements": "c6",
        "C7_Restrictions on internal movement": "c7",
        "C8_International travel controls": "c8",
        "Date": "date",
        "H1_Public information campaigns": "h1",
        "H2_Testing policy": "h2",
        "H3_Contact tracing": "h3",
        "H6_Facial Coverings": "h6",
        "CountryName": "country_name",
        "RegionName": "region_name"
    }, inplace=True)
    if 'ConfirmedCases' in df:
        df.rename(columns={'ConfirmedCases': 'cases'}, inplace=True)
    if 'ConfirmedDeaths' in df:
        df.rename(columns={'ConfirmedDeaths': 'deaths'}, inplace=True)
    return df


def apply_region_name_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df['region_name'].fillna('', inplace=True)
    return df


def apply_min(df: pd.DataFrame) -> pd.DataFrame:
    for e in df.items():
        name, series = e[0], e[1]
        if series.dtype == 'float64':
            series.clip(lower=0.0, inplace=True)
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = map_column_names(df)
    df = filter_columns(df)
    df = filter_countries(df)
    df = apply_region_name_cleaning(df)
    df = apply_min(df)
    return df
