import covid_io
import datetime
import calendar
import covid_pandas
import numpy as np


def sin(v, l):
    return np.sin(2 * np.pi * v / l)


def cos(v, l):
    return np.cos(2 * np.pi * v / l)


def resolve_days_in_year(d):
    return 366 if calendar.isleap(d.year) else 365


def resolve_days_in_month(d):
    return calendar.monthrange(d.year, d.month)[1]


covid_pandas.configure()
input_file_name = '03_data.csv'
output_file_name = '04_data.csv'

df = covid_io.read(input_file_name)

# Day of Week
df['date_day_of_week_class'] = df.date.apply(datetime.date.weekday)
df['date_day_of_year_sin'] = df.apply(lambda r: sin(r.date.timetuple().tm_yday, resolve_days_in_year(r.date)), axis=1)
df['date_day_of_year_cos'] = df.apply(lambda r: cos(r.date.timetuple().tm_yday, resolve_days_in_year(r.date)), axis=1)
df['date_day_of_month_sin'] = df.apply(lambda r: sin(r.date.day, resolve_days_in_month(r.date)), axis=1)
df['date_day_of_month_cos'] = df.apply(lambda r: cos(r.date.day, resolve_days_in_month(r.date)), axis=1)
df['date_day_of_week_sin'] = df.apply(lambda r: sin(r.date.weekday(), 6), axis=1)
df['date_day_of_week_cos'] = df.apply(lambda r: cos(r.date.weekday(), 6), axis=1)

day_of_first_known_infection = datetime.date(2019, 12, 1).toordinal()
df['date_days_since_start_of_calendar_n_nn'] = df.apply(lambda r: r.date.toordinal(), axis=1)
df['date_days_since_first_known_infection_n_nn'] = df.apply(lambda r: r.date.toordinal() - day_of_first_known_infection,
                                                            axis=1)

# Write the data
covid_io.write(df, output_file_name)
