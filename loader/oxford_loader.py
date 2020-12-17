import pandas as pd
import datetime


def date_parser(value: str) -> datetime.date:
    if len(value) == 8:
        return datetime.datetime.strptime(value, "%Y%m%d").date()
    else:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()


def load(fn: str) -> pd.DataFrame:
    na_values = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                 "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
                 "NULL", "NaN", "n/a", "nan", "null"]
    return pd.read_csv(f"data/{fn}",
                       parse_dates=['Date'],
                       date_parser=date_parser,
                       dtype={"RegionCode": str, "RegionName": str},
                       na_values=na_values,
                       keep_default_na=False,
                       error_bad_lines=False)
