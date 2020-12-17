import os

from loader import oxford_loader
import unittest
import datasets_constants


class TestOxfordLoader(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_loader(self):
        df = oxford_loader.load(datasets_constants.PATH_DATA_BASELINE)
        df.info()
        df = oxford_loader.load(datasets_constants.PATH_DATA_HISTORICAL)
        df.info()
        df = oxford_loader.load(datasets_constants.PATH_DATA_FUTURE)
        df.info()
