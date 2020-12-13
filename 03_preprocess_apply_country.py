import pycountry_convert
import covid_pandas
import covid_io
import pycountry
from workalendar.registry import registry


def resolve_continent(alpha3_country_class):
    alpha2_country_class = resolve_country_class(alpha3_country_class)
    if alpha2_country_class == 'TL':
        return 'UNKNOWN'
    else:
        return pycountry_convert.country_alpha2_to_continent_code(alpha2_country_class)


def resolve_country_class(x):
    if x == 'RKS':
        return 'XK'
    c = pycountry.countries.get(alpha_3=x)
    if c is None:
        return None
    else:
        return c.alpha_2


def is_working_day(r):
    working_registry = registry.get(resolve_country_class(r.country_class))
    working_calendar = default_calendar if working_registry is None else working_registry()
    return 1.0 if working_calendar.is_working_day(r.date) else 0.0


covid_pandas.configure()
input_file_name = '02_data.csv'
output_file_name = '03_data.csv'

# Add if it is a working day, based on the country data
df = covid_io.read(input_file_name)
default_calendar = registry.get('US')()
df['date_is_working_day_b_nn'] = df.apply(is_working_day, axis=1)
df['continent_class'] = df.country_class.apply(resolve_continent)
covid_io.write(df, output_file_name)
