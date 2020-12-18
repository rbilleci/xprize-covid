import os

import unittest
import datasets_constants
import oxford_loader


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
