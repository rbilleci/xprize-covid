import os

import pandas as pd
import unittest

from constants import PATH_DATA_BASELINE
from df_loader import load_ml_data

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_info_columns = 1000


class TestPipeline(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_oxford_data(self):
        train, validation, test = load_ml_data(PATH_DATA_BASELINE, 21, 21)
        train.info()
