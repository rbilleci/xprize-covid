import pandas as pd

check_for_nulls = sorted([])


def apply(df: pd.DataFrame) -> pd.DataFrame:
    # Mark columns with null values
    for e in df.items():
        name, series = e[0], e[1]
        if name in check_for_nulls:
            df[f"{name}_null"] = series.apply(lambda x: 1.0 if pd.isnull(x) else 0.0)
    return df
