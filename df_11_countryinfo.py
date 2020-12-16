import pandas as pd
import datasets
import datasets_working_days


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df['continent_code'] = df['country_name'].apply(lambda x: datasets.country_name_to_continent[x])
    df['is_working_day_today'] = df[['country_name', 'date']].apply(is_working_day_today, axis=1)
    df['is_working_day_tomorrow'] = df[['country_name', 'date']].apply(is_working_day_tomorrow, axis=1)
    df['is_working_day_yesterday'] = df[['country_name', 'date']].apply(is_working_day_yesterday, axis=1)
    return df


def is_working_day_today(x):
    return is_working_day(x['country_name'], x['date'].toordinal())


def is_working_day_tomorrow(x):
    return is_working_day(x['country_name'], x['date'].toordinal() + 1)


def is_working_day_yesterday(x):
    return is_working_day(x['country_name'], x['date'].toordinal() - 1)


def is_working_day(country_name: str, date_as_ordinal: int):
    if country_name in datasets_working_days.non_working_days:
        country_data = datasets_working_days.non_working_days[country_name]
        if date_as_ordinal not in country_data.keys():
            return 1.0
    return 0.0
