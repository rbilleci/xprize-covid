import unittest

from loader import oxford_loader, oxford_processor
from pipeline import df_00_splitter


class TestSplitter(unittest.TestCase):

    def test_splitter(self):
        df = oxford_processor.process(oxford_loader.load('OxCGRT_latest.csv'))
        sx, sy, sz = df_00_splitter.split(df, 21, 21)
