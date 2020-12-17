from loader import oxford_loader
import unittest

class TestOxfordLoader(unittest.TestCase):

    def test_loader(self):
        df = oxford_loader.load('OxCGRT_latest.csv')
        df.info()
        df = oxford_loader.load('2020-09-30_historical_ip.csv')
        df.info()
        df = oxford_loader.load('future_ip.csv')
        df.info()
