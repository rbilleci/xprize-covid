# Constants from oxford file
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

# Columns used for id of a row or grouping
INDEX_COLUMNS = [COUNTRY_NAME, REGION_NAME, DATE]
NPI_COLUMNS = [C1, C2, C3, C4, C5, C6, C7, C8, H1, H2, H3, H6]

LABEL = 'Label'
INCLUDE_CONFIRMED_CASES = False

# SPECIAL HANDLING
COLUMNS_TO_APPLY_NULL_MARKER = sorted([])

# OUTPUT COLUMNS
PREDICTED_NEW_CASES = 'PredictedDailyNewCases'
IS_SPECIALTY = 'IsSpecialty'
OUTPUT_COLUMNS = [COUNTRY_NAME, REGION_NAME, DATE, PREDICTED_NEW_CASES, IS_SPECIALTY]

# The columns we allow on read, filtering out those that do not belong
COLUMNS_ALLOWED_ON_READ = sorted([C1,
                                  C2,
                                  C3,
                                  C4,
                                  C5,
                                  C6,
                                  C7,
                                  C8,
                                  CONFIRMED_CASES,
                                  COUNTRY_NAME,
                                  # CONFIRMED_DEATHS,
                                  DATE,
                                  H1,
                                  H2,
                                  H3,
                                  H6,
                                  REGION_NAME])

PREDICTION_WINDOW_LIMIT = 100
