import datetime

DATE_SUBMISSION_CUTOFF = datetime.date(2020, 12, 22)  # the last known date we'll use to rely on reference data
DATE_LOWER_BOUND = datetime.date(2020, 1, 1)
DATE_UPPER_BOUND = datetime.date(2021, 12, 31)
DATE_ORDINAL_LOWER_BOUND = DATE_LOWER_BOUND.toordinal()
DATE_ORDINAL_UPPER_BOUND = DATE_UPPER_BOUND.toordinal()
LABEL_SCALING = 1e6

# dataset may change daily, but it seems we have to strip off two days of garbage data at the end...
DAYS_TO_STRIP_FROM_DATASET = 2

GEO_ID = 'GEO_ID'
COUNTRY_NAME = 'CountryName'
COUNTRY_CODE = 'CountryCode'
REGION_NAME = 'RegionName'
REGION_CODE = 'RegionCode'
DATE = 'Date'
C1 = "C1_School closing"
C2 = "C2_Workplace closing"
C3 = "C3_Cancel public events"
C4 = "C4_Restrictions on gatherings"
C5 = "C5_Close public transport"
C6 = "C6_Stay at home requirements"
C7 = "C7_Restrictions on internal movement"
C8 = "C8_International travel controls"
CONFIRMED_CASES = "ConfirmedCases"
CONFIRMED_DEATHS = "ConfirmedDeaths"
H1 = "H1_Public information campaigns"
H2 = "H2_Testing policy"
H3 = "H3_Contact tracing"
H6 = "H6_Facial Coverings"
PREDICTED_NEW_CASES = 'PredictedDailyNewCases'

INPUT_SCALE = {C1: 3.0,
               C2: 3.0,
               C3: 2.0,
               C4: 4.0,
               C5: 2.0,
               C6: 3.0,
               C7: 2.0,
               C8: 4.0,
               H1: 2.0,
               H2: 3.0,
               H3: 2.0,
               H6: 4.0,
               CONFIRMED_CASES: 1e8,  # 100 million
               CONFIRMED_DEATHS: 1e7,  # 10 million
               PREDICTED_NEW_CASES: LABEL_SCALING
               }
# For training data


PATH_DATA_BASELINE_RAW = 'data/data_baseline_raw.csv'
PATH_DATA_BASELINE = 'data/data_baseline.csv'
PATH_DATA_FUTURE = 'data/data_ip_future.csv'
PATH_DATA_HISTORICAL = 'data/data_ip_historic.csv'
PATH_DATA = 'data'

# For reference data
REFERENCE_COUNTRIES_AND_REGIONS = 'data/geo.csv'

# Columns augmented
CONTINENT = 'continent'
DAY_OF_WEEK = 'day_of_week'
DAY_OF_YEAR_SIN = 'day_of_year_sin'
DAY_OF_YEAR_COS = 'day_of_year_cos'
DAY_OF_MONTH_SIN = 'day_of_month_sin'
DAY_OF_MONTH_COS = 'day_of_month_cos'
DAY_OF_WEEK_SIN = 'day_of_week_sin'
DAY_OF_WEEK_COS = 'day_of_week_cos'
IS_WORKING_DAY_TODAY = 'is_working_day_today'
IS_WORKING_DAY_YESTERDAY = 'is_working_day_yesterday'
IS_WORKING_DAY_TOMORROW = 'is_working_day_tomorrow'
INDEX_COLUMNS = [COUNTRY_NAME, REGION_NAME, DATE]
NPI_COLUMNS = [C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6]
COLUMNS_TO_APPLY_NULL_MARKER = sorted([CONFIRMED_CASES])
IS_SPECIALTY = 'IsSpecialty'

OUTPUT_COLUMNS = INDEX_COLUMNS + [PREDICTED_NEW_CASES, IS_SPECIALTY]

# The columns we allow on read, filtering out those that do not belong
COLUMNS_ALLOWED_ON_READ = sorted(INDEX_COLUMNS + NPI_COLUMNS + [CONFIRMED_CASES] + [GEO_ID])