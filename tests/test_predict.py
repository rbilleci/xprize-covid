import os
import unittest

import predict as predict


class TestPredict(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_predict(self):
        # TODO: test with different date ranges to simulate unexpected problems
        predict.predict("2020-12-30", "2021-07-10", "data/future_ip.csv", "output_file.log")
