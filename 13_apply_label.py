import pandas as pd
import covid_io
import covid_pandas
import covid_scaler


def compute_label(dataset, fn_in, fn_out):
    df = covid_io.read(fn_in, dataset=dataset)
    df = compute_label_for_dataframe('confirmed_cases_n', df)

    # Scale the confirmed_cases label
    covid_scaler.scale(df, '_label', 0.0, 1e6, range_is_zero_to_one=False)

    # Remove information that may impact training!
    df.drop('confirmed_cases_n', axis=1, inplace=True)
    df.drop('confirmed_deaths_n', axis=1, inplace=True)

    covid_io.write(df, fn_out, dataset=dataset)


def compute_label_for_dataframe(series_name, df: pd.DataFrame):
    df.sort_values('date', inplace=True)
    grouped = df.groupby(df.geo_class)
    grouped = grouped.apply(lambda x: compute_label_for_group(series_name, x))
    return grouped.reset_index(drop=True)


def compute_label_for_group(source_series_name, group):
    group['_label'] = group[source_series_name].diff().fillna(0.0)
    return group


covid_pandas.configure()
input_file_name = '12_data.csv'
output_file_name = '13_data.csv'

compute_label('train', input_file_name, output_file_name)
compute_label('validation', input_file_name, output_file_name)
compute_label('test', input_file_name, output_file_name)
