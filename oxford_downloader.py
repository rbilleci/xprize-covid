import pathlib
import urllib.request
import datasets_constants

pathlib.Path(datasets_constants.PATH_DATA).mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve("https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv",
                           datasets_constants.PATH_DATA_BASELINE)
