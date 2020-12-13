import covid_io
import covid_splitter
import covid_pandas

covid_pandas.configure()

input_file_name = '04_data.csv'
output_file_name = '05_data.csv'

df = covid_io.read(input_file_name)

# Split the data into Training, Validation, and Test sets by date
df_train, df_validation, df_test = covid_splitter.split(df, 0.7, 0.2, 0.1)

# Write the data
covid_io.write(df_train, output_file_name, 'train')
covid_io.write(df_validation, output_file_name, 'validation')
covid_io.write(df_test, output_file_name, 'test')
