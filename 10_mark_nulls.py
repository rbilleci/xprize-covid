import pandas as pd
import covid_io
import covid_pandas
import covid_types

covid_pandas.configure()
input_file_name = '05_data.csv'
output_file_name = '10_data.csv'


def mark_null_columns(dataset, fn_in, fn_out):
    df = covid_io.read(fn_in, dataset)

    # Mark columns with null values
    for e in df.items():
        name, series = e[0], e[1]
        if covid_types.is_numeric(name):
            df[f"{name}_null"] = series.apply(lambda x: 1.0 if pd.isnull(x) else 0.0)

    # Write the data
    covid_io.write(df, fn_out, dataset)


mark_null_columns('train', input_file_name, output_file_name)
mark_null_columns('validation', input_file_name, output_file_name)
mark_null_columns('test', input_file_name, output_file_name)
