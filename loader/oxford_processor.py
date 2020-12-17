import pandas as pd

from datasets import datasets
from oxford_constants import *


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    for name in df.columns:
        if name not in COLUMNS_ALLOWED_ON_READ:
            df.drop(name, axis=1, inplace=True)
    return df


def filter_countries(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[COUNTRY_NAME].isin(datasets.ALLOWED_COUNTRIES)]
    return df


def apply_region_name_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df[REGION_NAME].fillna('', inplace=True)
    return df


def apply_min(df: pd.DataFrame) -> pd.DataFrame:
    for e in df.items():
        name, series = e[0], e[1]
        if series.dtype == 'float64':
            series.clip(lower=0.0, inplace=True)
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_columns(df)
    df = filter_countries(df)
    df = apply_region_name_cleaning(df)
    df = apply_min(df)
    return df
