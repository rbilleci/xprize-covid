import os

from pipeline import df_pipeline
import pandas as pd
import unittest

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_info_columns = 1000


class TestPipeline(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_oxford_data(self):
        train, validation, test = df_pipeline.process_for_training('data/baseline_data.csv', 21, 21)
        train.info()

    def test_historical_data(self):
        df = df_pipeline.process_for_prediction('data/2020-09-30_historical_ip.csv')
        df.info()

    def test_future_data(self):
        df = df_pipeline.process_for_prediction('data/future_ip.csv')
        df.info()
