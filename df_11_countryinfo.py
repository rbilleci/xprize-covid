import pandas as pd
import datasets
import datasets_working_days


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df['continent_code'] = df['country_name'].apply(lambda x: datasets.country_name_to_continent[x])
    df['is_working_day'] = df[['country_name', 'date']].apply(is_working_day, axis=1)
    return df


def is_working_day(x):
    if x['country_name'] in datasets_working_days.non_working_days:
        country_data = datasets_working_days.non_working_days[x['country_name']]
        if x['date'].toordinal() not in country_data.keys():
            return 1.0
    else:
        print(x['country_name'])
    return 0.0
