import pandas as pd
from datasets import datasets, datasets_working_days
from oxford_constants import CONTINENT, IS_WORKING_DAY_TODAY, IS_WORKING_DAY_TOMORROW, IS_WORKING_DAY_YESTERDAY, \
    COUNTRY_NAME, DATE


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df[CONTINENT] = df[COUNTRY_NAME].apply(lambda x: datasets.country_name_to_continent[x])
    df[IS_WORKING_DAY_TODAY] = df[[COUNTRY_NAME, DATE]].apply(is_working_day_today, axis=1)
    df[IS_WORKING_DAY_TOMORROW] = df[[COUNTRY_NAME, DATE]].apply(is_working_day_tomorrow, axis=1)
    df[IS_WORKING_DAY_YESTERDAY] = df[[COUNTRY_NAME, DATE]].apply(is_working_day_yesterday, axis=1)
    return df


def is_working_day_today(x):
    return is_working_day(x[COUNTRY_NAME], x[DATE].toordinal())


def is_working_day_tomorrow(x):
    return is_working_day(x[COUNTRY_NAME], x[DATE].toordinal() + 1)


def is_working_day_yesterday(x):
    return is_working_day(x[COUNTRY_NAME], x[DATE].toordinal() - 1)


def is_working_day(country_name: str, date_as_ordinal: int):
    if country_name in datasets_working_days.non_working_days:
        country_data = datasets_working_days.non_working_days[country_name]
        if date_as_ordinal not in country_data.keys():
            return 1.0
    return 0.0
