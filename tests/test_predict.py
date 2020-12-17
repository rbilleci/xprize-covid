import shutil
import os
import unittest

import predict as predict


class TestPredict(unittest.TestCase):

    def setUp(self) -> None:
        if os.path.exists('model'):
            shutil.rmtree('model')
        shutil.copytree('../model', 'model', dirs_exist_ok=True)

    def test_predict(self):
        predict.predict("2020-01-01", "2020-01-01", "input_file", "output_file")
