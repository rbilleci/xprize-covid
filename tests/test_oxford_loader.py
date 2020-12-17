import os

from loader import oxford_loader
import unittest


class TestOxfordLoader(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_loader(self):
        df = oxford_loader.load('data/baseline_data.csv')
        df.info()
        df = oxford_loader.load('data/2020-09-30_historical_ip.csv')
        df.info()
        df = oxford_loader.load('data/future_ip.csv')
        df.info()
