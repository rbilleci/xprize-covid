import calendar
import datetime
import pandas as pd
import numpy as np

twopi = 2 * np.pi


def sin(v, limit):
    return np.sin(twopi * v / limit)


def cos(v, limit):
    return np.cos(twopi * v / limit)


def resolve_days_in_month(d):
    return calendar.monthrange(d.year, d.month)[1]


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week'] = df.date.apply(datetime.date.weekday)
    df['day_of_year_sin'] = df.date.apply(lambda r: sin(r.timetuple().tm_yday, 365))
    df['day_of_year_cos'] = df.date.apply(lambda r: cos(r.timetuple().tm_yday, 365))
    df['day_of_month_sin'] = df.date.apply(lambda r: sin(r.day, resolve_days_in_month(r)))
    df['day_of_month_cos'] = df.date.apply(lambda r: cos(r.day, resolve_days_in_month(r)))
    df['day_of_week_sin'] = df.date.apply(lambda r: sin(r.weekday(), 6))
    df['day_of_week_cos'] = df.date.apply(lambda r: cos(r.weekday(), 6))
    return df
