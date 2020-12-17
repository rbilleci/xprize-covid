import pandas as pd
import pycountry_convert
from workalendar.registry import registry
from loader import oxford_loader
from datetime import timedelta
import datetime


def date_range(start_date, end_date, include_end_date):
    days = int((end_date - start_date).days)
    if include_end_date:
        days = days + 1
    for n in range(int(days)):
        yield start_date + timedelta(n)


df = oxford_loader.load('OxCGRT_latest.csv')

unmappable_country_alpha3_codes = sorted(['RKS'])
unmappable_country_alpha2_codes = sorted(['XX', 'TL'])
country_names = {}
country_name_to_continent = {}
country_name_to_alpha2_code = {}
country_name_to_alpha3_code = {}
region_name_to_region_code = {}

for index, row in df.iterrows():
    country_name = row['CountryName']
    country_names[country_name] = None
    country_alpha3_code = row['CountryCode']
    country_alpha2_code = 'XX' if country_alpha3_code in unmappable_country_alpha3_codes \
        else pycountry_convert.country_alpha3_to_country_alpha2(country_alpha3_code)
    continent_code = 'UNKNOWN' if country_alpha2_code in unmappable_country_alpha2_codes \
        else pycountry_convert.country_alpha2_to_continent_code(country_alpha2_code)
    region_name = None if pd.isnull(row['RegionName']) else row['RegionName']
    region_code = None if pd.isnull(row['RegionCode']) else row['RegionCode']
    country_name_to_continent[country_name] = continent_code
    country_name_to_alpha2_code[country_name] = country_alpha2_code
    country_name_to_alpha3_code[country_name] = country_alpha3_code
    if region_name is not None:
        region_name_to_region_code[region_name] = region_code

# generate working day information for each country
default_calendar = registry.get('US')()
start_date = datetime.date(2020, 1, 1)  # TODO: factor out start date across all code
end_date = datetime.date(2021, 12, 31)  # TODO: factor out end date across all code

non_working_dates = {}

for country_name in country_names.keys():
    working_registry = registry.get(country_name_to_alpha2_code[country_name])
    working_calendar = default_calendar if working_registry is None else working_registry()

    if country_name not in non_working_dates:
        non_working_dates[country_name] = {}

    for day in date_range(start_date, end_date, True):
        if not working_calendar.is_working_day(day):
            non_working_dates[country_name][day.toordinal()] = 1

print(non_working_dates)
# print out all the data
print(country_name_to_alpha2_code)
print(country_name_to_alpha3_code)
print(region_name_to_region_code)
print(country_name_to_continent)
