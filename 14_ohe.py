import pandas as pd
import covid_io
import covid_types
import numpy as np
import covid_pandas


def one_hot_encode(dataset, fn_in, fn_out):
    # Read Data
    df = covid_io.read(fn_in, dataset=dataset)

    # Process Series
    for e in df.items():
        name, series = e[0], e[1]
        if covid_types.is_boolean(name):
            df = df.join(pd.get_dummies(series.fillna(2.0).astype('int64'), prefix=f"{name}_class", dtype=np.float64))
            df = df.drop(name, axis=1)
        if covid_types.is_class(name):
            df = df.join(pd.get_dummies(series, prefix=name, dtype=np.float64))
            df = df.drop(name, axis=1)

    # Write outputs
    covid_io.write(df, fn_out, dataset=dataset)


covid_pandas.configure()
input_file_name = '13_data.csv'
output_file_name = '14_data.csv'

one_hot_encode('train', input_file_name, output_file_name)
one_hot_encode('validation', input_file_name, output_file_name)
one_hot_encode('test', input_file_name, output_file_name)
