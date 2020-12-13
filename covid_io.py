import pandas as pd
import datetime


def read(fn, dataset=None):
    fn = f"{fn}.gz"
    if dataset is None:
        pass
    else:
        fn = f"{dataset}_{fn}"
    return read_internal(fn, 'date', "%Y-%m-%d")


def read_oxford_data(fn):
    return read_internal(fn, 'Date', "%Y%m%d")


def read_internal(fn, date_field, date_format):
    na_values = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                 "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
                 "NULL", "NaN", "n/a", "nan", "null"]
    return pd.read_csv(f"data/{fn}",
                       parse_dates=[date_field],
                       date_parser=lambda x: datetime.datetime.strptime(x, date_format).date(),
                       dtype={"RegionCode": str, "RegionName": str},
                       na_values=na_values,
                       keep_default_na=False,
                       error_bad_lines=False)


def write(df, fn, dataset=None):
    # Perform a sort of column values
    df = df.reindex(sorted(df.columns), axis=1)
    df.info()
    # write the final
    fn = f"{fn}.gz"
    if dataset is None:
        pass
    else:
        fn = f"{dataset}_{fn}"
    df.to_csv(f"data/{fn}", index=False, compression='gzip')
