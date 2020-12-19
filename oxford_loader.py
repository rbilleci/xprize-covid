import datetime

import pandas as pd

from constants import *

# Dataframe of Geo data
df_geos = pd.read_csv(REFERENCE_COUNTRIES_AND_REGIONS)
df_geos[REGION_NAME] = df_geos[REGION_NAME].fillna('')
df_geos[GEO_ID] = df_geos[COUNTRY_NAME] + df_geos[REGION_NAME]


def load(fn: str) -> pd.DataFrame:
    na_values = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                 "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
                 "NULL", "NaN", "n/a", "nan", "null"]
    df = pd.read_csv(fn,
                     parse_dates=[DATE],
                     date_parser=date_parser,  # move to iso--???
                     dtype={REGION_CODE: str, REGION_NAME: str},
                     na_values=na_values,
                     keep_default_na=False,
                     error_bad_lines=False)
    df = add_geo_id(df)
    df = filter_by_geo_id(df)
    df = filter_by_columns(df)
    df = clip_values(df)
    return df


def add_geo_id(df: pd.DataFrame) -> pd.DataFrame:
    df[REGION_NAME] = df[REGION_NAME].fillna('')
    df[GEO_ID] = df[COUNTRY_NAME] + df[REGION_NAME]
    return df


def filter_by_geo_id(df: pd.DataFrame) -> pd.DataFrame:
    allowed_geo_ids = df_geos[GEO_ID].values.tolist()
    return df[df[GEO_ID].isin(allowed_geo_ids)]


def filter_by_columns(df: pd.DataFrame) -> pd.DataFrame:
    for name in df.columns:
        if name not in COLUMNS_ALLOWED_ON_READ:
            df.drop(name, axis=1, inplace=True)
    return df


def clip_values(df: pd.DataFrame) -> pd.DataFrame:
    for e in df.items():
        name, series = e[0], e[1]
        if series.dtype == 'float64':
            series.clip(lower=0.0, inplace=True)
    return df


def date_parser(value: str) -> datetime.date:
    if len(value) == 8:
        return datetime.datetime.strptime(value, "%Y%m%d").date()
    else:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
