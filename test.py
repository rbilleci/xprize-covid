import oxford_loader
from datasets_constants import PATH_DATA_BASELINE_RAW, PATH_DATA_BASELINE
from oxford_constants import CONFIRMED_CASES
import pandas as pd

df = oxford_loader.load(PATH_DATA_BASELINE)
pd.options.display.max_columns = 4
pd.options.display.max_rows = 1000
pd.options.display.max_info_columns = 1000

df.info()
m = df[df[CONFIRMED_CASES].isnull()]
print(m.head(10))
