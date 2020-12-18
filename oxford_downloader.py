import pathlib
import urllib.request

import oxford_loader
from datasets_constants import PATH_DATA_BASELINE, PATH_DATA_BASELINE_RAW, PATH_DATA
from oxford_constants import GEO_ID


def download():
    # First, download the file
    pathlib.Path(PATH_DATA).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv",
                               PATH_DATA_BASELINE_RAW)
    # Second, perform preprocessing on the file to remove any data
    # we are not using
    df = oxford_loader.load(PATH_DATA_BASELINE_RAW)
    df = df.drop(GEO_ID, axis=1)
    df.to_csv(PATH_DATA_BASELINE, index=False)


download()
