import calendar
import datetime
import pandas as pd
import numpy as np
from oxford_constants import *

twopi = 2 * np.pi


def sin(v, limit):
    return np.sin(twopi * v / limit)


def cos(v, limit):
    return np.cos(twopi * v / limit)


def resolve_days_in_month(d):
    return calendar.monthrange(d.year, d.month)[1]


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df[DAY_OF_WEEK] = df[DATE].apply(datetime.date.weekday)
    df[DAY_OF_WEEK_SIN] = df[DATE].apply(lambda r: sin(r.weekday(), 6))
    df[DAY_OF_WEEK_COS] = df[DATE].apply(lambda r: cos(r.weekday(), 6))
    df[DAY_OF_YEAR_SIN] = df[DATE].apply(lambda r: sin(r.timetuple().tm_yday, 365))
    df[DAY_OF_YEAR_COS] = df[DATE].apply(lambda r: cos(r.timetuple().tm_yday, 365))
    df[DAY_OF_MONTH_SIN] = df[DATE].apply(lambda r: sin(r.day, resolve_days_in_month(r)))
    df[DAY_OF_MONTH_COS] = df[DATE].apply(lambda r: cos(r.day, resolve_days_in_month(r)))
    return df
