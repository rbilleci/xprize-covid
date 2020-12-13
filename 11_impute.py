import pandas as pd
import covid_io
import covid_types
import covid_pandas


def impute(dataset, fn_in, fn_out):
    df = covid_io.read(fn_in, dataset)
    df = impute_dataframe(df)
    covid_io.write(df, fn_out, dataset)


def impute_dataframe(df: pd.DataFrame):
    df.sort_values('date', inplace=True)
    grouped = df.groupby(df.geo_class)
    grouped = grouped.apply(impute_group)
    return grouped.reset_index(drop=True)


def impute_group(group):
    return group.apply(impute_series)


def impute_series(series: pd.Series):
    if covid_types.is_numeric(series.name):
        if pd.isnull(series.iloc[0]):
            series.iloc[0] = 0.0  # Set the initial value to zero, if it is undefined
        return series.interpolate(method='linear')
    else:
        return series


covid_pandas.configure()
input_file_name = '10_data.csv'
output_file_name = '11_data.csv'
impute('train', input_file_name, output_file_name)
impute('validation', input_file_name, output_file_name)
impute('test', input_file_name, output_file_name)
