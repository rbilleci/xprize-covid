import pandas as pd

from datasets_constants import DATE_ORDINAL_UPPER_BOUND, DATE_ORDINAL_LOWER_BOUND, LABEL_SCALING
from oxford_constants import C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6, CONFIRMED_CASES, CONFIRMED_DEATHS, DATE, \
    LABEL

scales = {C1: 3.0,
          C2: 3.0,
          C3: 2.0,
          C4: 4.0,
          C5: 2.0,
          C6: 3.0,
          C7: 2.0,
          C8: 4.0,
          H1: 2.0,
          H2: 3.0,
          H3: 2.0,
          H6: 4.0,
          CONFIRMED_CASES: 1e8,  # 100 million
          CONFIRMED_DEATHS: 1e7,  # 10 million
          LABEL: LABEL_SCALING
          }


def apply(df: pd.DataFrame) -> pd.DataFrame:
    # Scale Standard Numeric Values
    for e in df.items():
        name, series = e[0], e[1]
        if name in scales.keys():
            df[name] = df[name].apply(lambda x: scale_value(x, 0, scales.get(name)))
        # TODO: cleanup
        if name.endswith('_sin') or name.endswith('_cos') or name.endswith('_SIN') or name.endswith('_COS'):
            df[name] = df[name].apply(lambda x: scale_value(x, -1.0, 1.0))
    # Scale the date
    df[DATE] = df[DATE].apply(lambda x: scale_value(x.toordinal(), DATE_ORDINAL_LOWER_BOUND, DATE_ORDINAL_UPPER_BOUND))
    return df


def scale_value(x, min_value, max_value):
    x = (x - min_value) / (max_value - min_value)
    x = max(0.0, x)
    x = min(1.0, x)
    return x
