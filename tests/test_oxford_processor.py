import os

from loader import oxford_loader, oxford_processor
import covid_constants
import unittest


class TestOxfordProcessor(unittest.TestCase):

    def setUp(self) -> None:
        print(os.getcwd())
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_oxford_processor(self):
        df = oxford_processor.process(oxford_loader.load(covid_constants.path_data_baseline))
        df.info()
        df = oxford_processor.process(oxford_loader.load(covid_constants.path_data_historical))
        df.info()
        df = oxford_processor.process(oxford_loader.load(covid_constants.path_data_future))
        df.info()
