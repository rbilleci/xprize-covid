import datetime

"""
When considering the date/time we start predicting from...

We'll have the final reference file on the 22nd. It should have values for the 21st, so the official date we begin
predicting values from will be on the 22nd.
"""
DATE_SUBMISSION_CUTOFF = datetime.date(2020, 12, 22)  # the last known date we'll use to rely on reference data

DATE_LOWER_BOUND = datetime.date(2020, 1, 1)
DATE_UPPER_BOUND = datetime.date(2021, 12, 31)
DATE_ORDINAL_LOWER_BOUND = DATE_LOWER_BOUND.toordinal()
DATE_ORDINAL_UPPER_BOUND = DATE_UPPER_BOUND.toordinal()

# dataset may change daily, but it seems we have to strip off two days of garbage data at the end...
DAYS_TO_STRIP_FROM_DATASET = 2

USE_CASES_PER_100K = False

MA_WINDOW_A = 3
MA_WINDOW_B = 7
MA_WINDOW_C = 21

SUFFIX_MA_A = "_ma_a"
SUFFIX_MA_B = "_ma_b"
SUFFIX_MA_C = "_ma_c"

# Values we must calculate during prediction
PREDICTED_NEW_CASES = 'PredictedDailyNewCases'
PREDICTED_NEW_CASES_MA_A = PREDICTED_NEW_CASES + SUFFIX_MA_A
PREDICTED_NEW_CASES_MA_B = PREDICTED_NEW_CASES + SUFFIX_MA_B
PREDICTED_NEW_CASES_MA_C = PREDICTED_NEW_CASES + SUFFIX_MA_C
CONFIRMED_CASES = "ConfirmedCases"
CONFIRMED_CASES_MA_A = CONFIRMED_CASES + SUFFIX_MA_A
CONFIRMED_CASES_MA_B = CONFIRMED_CASES + SUFFIX_MA_B
CONFIRMED_CASES_MA_C = CONFIRMED_CASES + SUFFIX_MA_C

# NPI COLUMNS
C1 = "C1_School closing"
C1_MA_A = C1 + SUFFIX_MA_A
C1_MA_B = C1 + SUFFIX_MA_B
C1_MA_C = C1 + SUFFIX_MA_C
C2 = "C2_Workplace closing"
C2_MA_A = C2 + SUFFIX_MA_A
C2_MA_B = C2 + SUFFIX_MA_B
C2_MA_C = C2 + SUFFIX_MA_C
C3 = "C3_Cancel public events"
C3_MA_A = C3 + SUFFIX_MA_A
C3_MA_B = C3 + SUFFIX_MA_B
C3_MA_C = C3 + SUFFIX_MA_C
C4 = "C4_Restrictions on gatherings"
C4_MA_A = C4 + SUFFIX_MA_A
C4_MA_B = C4 + SUFFIX_MA_B
C4_MA_C = C4 + SUFFIX_MA_C
C5 = "C5_Close public transport"
C5_MA_A = C5 + SUFFIX_MA_A
C5_MA_B = C5 + SUFFIX_MA_B
C5_MA_C = C5 + SUFFIX_MA_C
C6 = "C6_Stay at home requirements"
C6_MA_A = C6 + SUFFIX_MA_A
C6_MA_B = C6 + SUFFIX_MA_B
C6_MA_C = C6 + SUFFIX_MA_C
C7 = "C7_Restrictions on internal movement"
C7_MA_A = C7 + SUFFIX_MA_A
C7_MA_B = C7 + SUFFIX_MA_B
C7_MA_C = C7 + SUFFIX_MA_C
C8 = "C8_International travel controls"
C8_MA_A = C8 + SUFFIX_MA_A
C8_MA_B = C8 + SUFFIX_MA_B
C8_MA_C = C8 + SUFFIX_MA_C
H1 = "H1_Public information campaigns"
H1_MA_A = H1 + SUFFIX_MA_A
H1_MA_B = H1 + SUFFIX_MA_B
H1_MA_C = H1 + SUFFIX_MA_C
H2 = "H2_Testing policy"
H2_MA_A = H2 + SUFFIX_MA_A
H2_MA_B = H2 + SUFFIX_MA_B
H2_MA_C = H2 + SUFFIX_MA_C
H3 = "H3_Contact tracing"
H3_MA_A = H3 + SUFFIX_MA_A
H3_MA_B = H3 + SUFFIX_MA_B
H3_MA_C = H3 + SUFFIX_MA_C
H6 = "H6_Facial Coverings"
H6_MA_A = H6 + SUFFIX_MA_A
H6_MA_B = H6 + SUFFIX_MA_B
H6_MA_C = H6 + SUFFIX_MA_C

# Other columns
GEO_ID = 'GEO_ID'
COUNTRY_NAME = 'CountryName'
COUNTRY_CODE = 'CountryCode'
REGION_NAME = 'RegionName'
REGION_CODE = 'RegionCode'
DATE = 'Date'
CONFIRMED_DEATHS = "ConfirmedDeaths"

# Supplemental columns
POPULATION_DENSITY = 'PopulationDensity'
POPULATION = 'Population'

TRAINING_DATA_START_DATE = datetime.date(2020, 2, 1)  # don't train on earlier data

INPUT_SCALE = {
    C1: 3.0, C1_MA_A: 3.0, C1_MA_B: 3.0, C1_MA_C: 3.0,
    C2: 3.0, C2_MA_A: 3.0, C2_MA_B: 3.0, C2_MA_C: 3.0,
    C3: 2.0, C3_MA_A: 2.0, C3_MA_B: 2.0, C3_MA_C: 2.0,
    C4: 4.0, C4_MA_A: 4.0, C4_MA_B: 4.0, C4_MA_C: 4.0,
    C5: 2.0, C5_MA_A: 2.0, C5_MA_B: 2.0, C5_MA_C: 2.0,
    C6: 3.0, C6_MA_A: 3.0, C6_MA_B: 3.0, C6_MA_C: 3.0,
    C7: 2.0, C7_MA_A: 2.0, C7_MA_B: 2.0, C7_MA_C: 2.0,
    C8: 4.0, C8_MA_A: 4.0, C8_MA_B: 4.0, C8_MA_C: 4.0,
    H1: 2.0, H1_MA_A: 2.0, H1_MA_B: 2.0, H1_MA_C: 2.0,
    H2: 3.0, H2_MA_A: 3.0, H2_MA_B: 3.0, H2_MA_C: 3.0,
    H3: 2.0, H3_MA_A: 2.0, H3_MA_B: 2.0, H3_MA_C: 2.0,
    H6: 4.0, H6_MA_A: 4.0, H6_MA_B: 4.0, H6_MA_C: 4.0,
    # CONFIRMED CASES
    CONFIRMED_CASES: 1e8,  # 100 million,
    CONFIRMED_CASES_MA_A: 1e8,  # 100 million
    CONFIRMED_CASES_MA_B: 1e8,  # 100 million
    CONFIRMED_CASES_MA_C: 1e8,  # 100 million
    # NEW CASES
    PREDICTED_NEW_CASES: 1e6,
    PREDICTED_NEW_CASES_MA_A: 1e6,  # 100 million
    PREDICTED_NEW_CASES_MA_B: 1e6,  # 100 million
    PREDICTED_NEW_CASES_MA_C: 1e6,  # 100 million
    POPULATION: 1e10,
    POPULATION_DENSITY: 1e5  # max 100K
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
COLUMNS_TO_APPLY_NULL_MARKER = sorted([])  # don't use this for now
IS_SPECIALTY = 'IsSpecialty'

OUTPUT_COLUMNS = INDEX_COLUMNS + [PREDICTED_NEW_CASES, IS_SPECIALTY]

# The columns we allow on read, filtering out those that do not belong
COLUMNS_ALLOWED_ON_READ = sorted(INDEX_COLUMNS + NPI_COLUMNS + [CONFIRMED_CASES] + [GEO_ID])
