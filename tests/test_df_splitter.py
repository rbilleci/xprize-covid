import os
import unittest
import covid_constants
from loader import oxford_loader, oxford_processor
from pipeline import df_00_splitter


class TestSplitter(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_splitter(self):
        df = oxford_processor.process(oxford_loader.load(covid_constants.PATH_DATA_BASELINE))
        sx, sy, sz = df_00_splitter.split(df, 21, 21)
