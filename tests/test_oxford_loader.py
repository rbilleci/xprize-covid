import os

from loader import oxford_loader
import unittest
import covid_constants


class TestOxfordLoader(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_loader(self):
        df = oxford_loader.load(covid_constants.path_data_baseline)
        df.info()
        df = oxford_loader.load(covid_constants.path_data_historical)
        df.info()
        df = oxford_loader.load(covid_constants.path_data_future)
        df.info()
