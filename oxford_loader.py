import datetime

import pandas as pd

from datasets_constants import REFERENCE_COUNTRIES_AND_REGIONS
from oxford_constants import DATE, REGION_NAME, REGION_CODE, COUNTRY_NAME, GEO_ID, COLUMNS_ALLOWED_ON_READ


def load(fn: str, load_for_prediction=False) -> pd.DataFrame:
    na_values = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                 "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
                 "NULL", "NaN", "n/a", "nan", "null"]
    df = pd.read_csv(fn,
                     parse_dates=[DATE],
                     date_parser=date_parser,
                     dtype={REGION_CODE: str, REGION_NAME: str},
                     na_values=na_values,
                     keep_default_na=False,
                     error_bad_lines=False)
    df = apply_column_filter(df)
    df = apply_geo_grouping(df)
    df = apply_geo_filter(df, load_for_prediction)
    df = apply_min(df)
    return df


def apply_geo_grouping(df: pd.DataFrame) -> pd.DataFrame:
    df[REGION_NAME] = df[REGION_NAME].fillna('')
    df[GEO_ID] = df[COUNTRY_NAME] + df[REGION_NAME]
    return df


def apply_geo_filter(df: pd.DataFrame, load_for_prediction: bool) -> pd.DataFrame:
    if load_for_prediction:
        df_geos = pd.read_csv(REFERENCE_COUNTRIES_AND_REGIONS)
        df_geos[REGION_NAME] = df_geos[REGION_NAME].fillna('')
        df_geos[GEO_ID] = df_geos[COUNTRY_NAME] + df_geos[REGION_NAME]
        allowed_geo_ids = df_geos[GEO_ID].values.tolist()
        df = df[df[GEO_ID].isin(allowed_geo_ids)]
    return df


def apply_column_filter(df: pd.DataFrame) -> pd.DataFrame:
    for name in df.columns:
        if name not in COLUMNS_ALLOWED_ON_READ:
            df.drop(name, axis=1, inplace=True)
    return df


def apply_min(df: pd.DataFrame) -> pd.DataFrame:
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
