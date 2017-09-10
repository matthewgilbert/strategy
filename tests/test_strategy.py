from strategy.portfolios import QuarterlyPortfolio
import pandas as pd
import strategy.strategy as strat
import unittest
import os
from pandas.util.testing import assert_frame_equal, assert_series_equal, \
    assert_dict_equal


class TestBlotter(unittest.TestCase):

    def setUp(self):
        cdir = os.path.dirname(__file__)
        self.marketdata = os.path.join(cdir, 'marketdata')
        self.metadata = os.path.join(cdir, 'marketdata',
                                     'instrument_meta.json')

    def tearDown(self):
        pass

    def test_dummy(self):
        self.assertEqual(3, 3)
