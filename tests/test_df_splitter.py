import os
import unittest

from loader import oxford_loader, oxford_processor
from pipeline import df_00_splitter


class TestSplitter(unittest.TestCase):

    def setUp(self) -> None:
        if os.getcwd().endswith('/tests'):
            os.chdir("../")

    def test_splitter(self):
        df = oxford_processor.process(oxford_loader.load('baseline_data.csv'))
        sx, sy, sz = df_00_splitter.split(df, 21, 21)
