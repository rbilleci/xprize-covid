import os

from loader import oxford_loader, oxford_processor
import unittest


class TestOxfordProcessor(unittest.TestCase):

    def setUp(self) -> None:
        print(os.getcwd())
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_oxford_processor(self):
        df = oxford_processor.process(oxford_loader.load('baseline_data.csv'))
        df.info()
        df = oxford_processor.process(oxford_loader.load('2020-09-30_historical_ip.csv'))
        df.info()
        df = oxford_processor.process(oxford_loader.load('future_ip.csv'))
        df.info()
