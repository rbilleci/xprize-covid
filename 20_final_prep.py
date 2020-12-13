import covid_io
import covid_pandas


def remove_unused_columns(dataset, fn_in, fn_out):
    df = covid_io.read(fn_in, dataset=dataset)
    df = df.drop('date', axis=1)
    covid_io.write(df, fn_out, dataset=dataset)


covid_pandas.configure()
input_file_name = '14_data.csv'
output_file_name = '20_data.csv'

remove_unused_columns('train', input_file_name, output_file_name)
remove_unused_columns('validation', input_file_name, output_file_name)
remove_unused_columns('test', input_file_name, output_file_name)
