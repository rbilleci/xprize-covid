import pandas as pd
import datetime

scales = {'c1': 3.0,
          'c2': 3.0,
          'c3': 2.0,
          'c4': 4.0,
          'c5': 2.0,
          'c6': 3.0,
          'c7': 2.0,
          'c8': 4.0,
          'h1': 2.0,
          'h2': 3.0,
          'h3': 2.0,
          'h6': 4.0,
          'cases': 1e9,  # 1 billion
          'deaths': 1e8,  # 100 million
          '_label': 1e6
          }

date_ordinal_min = datetime.date(2020, 1, 1).toordinal()
date_ordinal_max = datetime.date(2021, 12, 31).toordinal()


def apply(df: pd.DataFrame) -> pd.DataFrame:
    # Scale Standard Numeric Values
    for e in df.items():
        name, series = e[0], e[1]
        if name in scales.keys():
            df[name] = df[name].apply(lambda x: scale_value(x, 0, scales.get(name)))
        if name.endswith('_sin') or name.endswith('_cos'):
            df[name] = df[name].apply(lambda x: scale_value(x, -1.0, 1.0))
    # Scale the date
    df['date'] = df['date'].apply(lambda x: scale_value(x.toordinal(), date_ordinal_min, date_ordinal_max))
    return df


def scale_value(x, min_value, max_value):
    x = (x - min_value) / (max_value - min_value)
    x = max(0.0, x)
    x = min(1.0, x)
    return x
