# For training data
import datetime

PATH_DATA_BASELINE_RAW = 'data/data_baseline_raw.csv'
PATH_DATA_BASELINE = 'data/data_baseline.csv'
PATH_DATA_FUTURE = 'data/future_ip.csv'
PATH_DATA_HISTORICAL = 'data/2020-09-30_historical_ip.csv'
PATH_DATA = 'data'

# For reference data
REFERENCE_COUNTRIES_AND_REGIONS = 'data/reference_countries_and_regions.csv'

DATE_LOWER_BOUND = datetime.date(2020, 1, 1)
DATE_UPPER_BOUND = datetime.date(2021, 12, 31)
DATE_ORDINAL_LOWER_BOUND = DATE_LOWER_BOUND.toordinal()
DATE_ORDINAL_UPPER_BOUND = DATE_UPPER_BOUND.toordinal()
LABEL_SCALING = 1e6

# dataset may change daily, but it seems we have to strip off two days of garbage data at the end...
DAYS_TO_STRIP_FROM_DATASET = 2
