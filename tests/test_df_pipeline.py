import os

from pipeline import df_pipeline
import pandas as pd
import unittest
import datasets_constants

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_info_columns = 1000


class TestPipeline(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_oxford_data(self):
        train, validation, test = df_pipeline.get_datasets_for_training(datasets_constants.PATH_DATA_BASELINE, 21, 21)
        train.info()
