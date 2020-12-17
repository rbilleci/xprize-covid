import pathlib
import urllib.request
import covid_constants

pathlib.Path(covid_constants.PATH_DATA).mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve("https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv",
                           covid_constants.PATH_DATA_BASELINE)
