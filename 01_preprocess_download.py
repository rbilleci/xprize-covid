import pathlib
import urllib.request

output_data_directory = 'data'
output_file_name = '01_data.csv'

pathlib.Path(output_data_directory).mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve("https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv",
                           f"{output_data_directory}/{output_file_name}")
