import pandas as pd
from oxford_constants import *


def apply(df: pd.DataFrame) -> pd.DataFrame:
    for e in df.items():
        name, series = e[0], e[1]
        if name in COLUMNS_TO_APPLY_NULL_MARKER:
            df[f"{name}_null"] = series.apply(lambda x: 1.0 if pd.isnull(x) else 0.0)
    return df
