import pandas as pd


def configure():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.max_info_columns = 1000
