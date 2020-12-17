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
        df = oxford_processor.process(oxford_loader.load(covid_constants.PATH_DATA_BASELINE))
        df.info()
        df = oxford_processor.process(oxford_loader.load(covid_constants.PATH_DATA_HISTORICAL))
        df.info()
        df = oxford_processor.process(oxford_loader.load(covid_constants.PATH_DATA_FUTURE))
        df.info()
