import pandas as pd
import numpy as np
from datasets import datasets
from oxford_constants import *


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df = ohe(df, COUNTRY_NAME, datasets.ALLOWED_COUNTRIES)
    df = ohe(df, REGION_NAME, datasets.region_name_to_region_code.keys())
    df = ohe(df, CONTINENT, datasets.country_name_to_continent.values())
    df = ohe(df, DAY_OF_WEEK, range(0, 7))
    return df


def ohe(df: pd.DataFrame, column: str, values) -> pd.DataFrame:
    df = df.join(pd.get_dummies(df[column], prefix=column, dtype=np.float64))
    for value in values:
        class_name = f"{column}_{value}"
        if class_name not in df.keys():
            df[class_name] = 0.0
    df = df.drop(column, axis=1)
    return df
