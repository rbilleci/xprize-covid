import os
import unittest

import predict as predict


class TestPredict(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_predict(self):
        predict.predict("2020-01-01", "2020-01-01", "data/future_ip.csv", "output_file")
