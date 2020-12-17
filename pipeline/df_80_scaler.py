import pandas as pd
import datetime
from oxford_constants import *

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
          CASES: 1e9,  # 1 billion
          DEATHS: 1e8,  # 100 million
          LABEL: 1e6
          }

date_ordinal_min = datetime.date(2020, 1, 1).toordinal()
date_ordinal_max = datetime.date(2021, 12, 31).toordinal()


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
    df[DATE] = df[DATE].apply(lambda x: scale_value(x.toordinal(), date_ordinal_min, date_ordinal_max))
    return df


def scale_value(x, min_value, max_value):
    x = (x - min_value) / (max_value - min_value)
    x = max(0.0, x)
    x = min(1.0, x)
    return x
