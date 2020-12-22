from datetime import timedelta

import pandas as pd
import pycountry_convert
from workalendar.registry import registry

import oxford_loader
from constants import *


def date_range(start_date, end_date, include_end_date):
    days = int((end_date - start_date).days)
    if include_end_date:
        days = days + 1
    for n in range(int(days)):
        yield start_date + timedelta(n)


df_countries = pd.read_csv(f"../{REFERENCE_COUNTRIES_AND_REGIONS}")
df_countries.fillna('', inplace=True)
country_names = df_countries[COUNTRY_NAME].unique().tolist()

df = oxford_loader.load(f"../{PATH_DATA_BASELINE}")
unmappable_country_alpha3_codes = sorted(['RKS'])
unmappable_country_alpha2_codes = sorted(['XX', 'TL'])
country_name_to_continent = {}
country_name_to_alpha2_code = {}
country_name_to_alpha3_code = {}
region_name_to_region_code = {}

for index, row in df.iterrows():
    country_name = row[COUNTRY_NAME]
    if country_name in country_names:
        country_alpha3_code = row[COUNTRY_CODE]
        country_alpha2_code = 'XX' if country_alpha3_code in unmappable_country_alpha3_codes \
            else pycountry_convert.country_alpha3_to_country_alpha2(country_alpha3_code)
        continent_code = 'UNKNOWN' if country_alpha2_code in unmappable_country_alpha2_codes \
            else pycountry_convert.country_alpha2_to_continent_code(country_alpha2_code)
        region_name = None if pd.isnull(row[REGION_NAME]) else row[REGION_NAME]
        region_code = None if pd.isnull(row[REGION_CODE]) else row[REGION_CODE]
        country_name_to_continent[country_name] = continent_code
        country_name_to_alpha2_code[country_name] = country_alpha2_code
        country_name_to_alpha3_code[country_name] = country_alpha3_code
        if region_name is not None:
            region_name_to_region_code[region_name] = region_code

# generate working day information for each country
default_calendar = registry.get('US')()
start_date = DATE_LOWER_BOUND
end_date = DATE_UPPER_BOUND

non_working_dates = {}

for country_name in country_names:
    working_registry = registry.get(country_name_to_alpha2_code[country_name])
    working_calendar = default_calendar if working_registry is None else working_registry()

    if country_name not in non_working_dates:
        non_working_dates[country_name] = {}

    for day in date_range(start_date, end_date, True):
        if not working_calendar.is_working_day(day):
            non_working_dates[country_name][day.toordinal()] = 1

# print out all the data
print(non_working_dates)
print(country_name_to_alpha2_code)
print(country_name_to_alpha3_code)
print(region_name_to_region_code)
print(country_name_to_continent)
print(country_names)
