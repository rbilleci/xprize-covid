import pandas as pd
import datetime
from oxford_constants import *


def date_parser(value: str) -> datetime.date:
    if len(value) == 8:
        return datetime.datetime.strptime(value, "%Y%m%d").date()
    else:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()


def load(fn: str) -> pd.DataFrame:
    na_values = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                 "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
                 "NULL", "NaN", "n/a", "nan", "null"]
    return pd.read_csv(fn,
                       parse_dates=[DATE],
                       date_parser=date_parser,
                       dtype={REGION_CODE: str, REGION_NAME: str},
                       na_values=na_values,
                       keep_default_na=False,
                       error_bad_lines=False)
