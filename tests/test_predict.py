import os
import unittest
import covid_constants
import predict as predict


class TestPredict(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_predict(self):
        # test a single day in the past
        self.predict_for_range("2020-10-01", "2020-10-01")
        # test two days in the past
        self.predict_for_range("2020-10-01", "2020-10-02")
        # test a single day in the future
        self.predict_for_range("2020-12-30", "2020-12-30")
        self.predict_for_range("2021-01-01", "2021-01-01")
        # test a two days in the future, with the calendar year switchover
        self.predict_for_range("2020-12-31", "2021-01-01")
        # test the range we will use for the challenge
        self.predict_for_range("2020-12-22", "2021-06-19")

    @staticmethod
    def predict_for_range(start_date, end_date):
        predict.predict(start_date, end_date, covid_constants.PATH_DATA_FUTURE,
                        f"tests/data_test_predictions_{start_date}_{end_date}.log")
