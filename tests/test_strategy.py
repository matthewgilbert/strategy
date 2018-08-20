import strategy.strategy as strat
import pandas as pd
import numpy as np
import unittest
import os
from pandas.util.testing import assert_frame_equal, assert_series_equal, \
    assert_index_equal
from . import util

# suppress missing price data warnings
strat.WARNINGS = "ignore"


class TestExpiryPortfolio(unittest.TestCase):

    def setUp(self):
        cdir = os.path.dirname(__file__)
        self.marketdata = os.path.join(cdir, 'marketdata')
        self.metadata = os.path.join(cdir, 'marketdata',
                                     'instrument_meta.json')
        self.CAPITAL = 1000000
        self.sd = pd.Timestamp("2015-01-02")
        self.ed = pd.Timestamp("2015-03-23")

    def tearDown(self):
        pass

    def assert_simulation_equal(self, sim1, sim2):
        self.assertEquals(sim1._fields, sim2._fields)
        assert_frame_equal(sim1.holdings, sim2.holdings, check_exact=False)
        assert_frame_equal(sim1.trades, sim2.trades, check_exact=False)
        assert_series_equal(sim1.pnl, sim2.pnl, check_exact=False)

    def test_rebalance_dates(self):
        # each contract rolling individually, same offset
        exposures = util.make_exposures(
            ["ES", "TY"], self.metadata, self.marketdata
        )
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL
        )
        rebal_dates = portfolio.rebalance_dates()
        exp_rebal_dates = pd.DatetimeIndex(["2015-01-02", "2015-02-24",
                                            "2015-03-17"])
        assert_index_equal(rebal_dates, exp_rebal_dates)

        # rolling all monthly contracts together, same offset
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL, all_monthly=True
        )
        rebal_dates = portfolio.rebalance_dates()
        exp_rebal_dates = pd.DatetimeIndex(["2015-01-02", "2015-02-24"])
        assert_index_equal(rebal_dates, exp_rebal_dates)

        # rolling each contract individually, different offset
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL,
            offset={"ES": -3, "TY": -4}
        )
        rebal_dates = portfolio.rebalance_dates()
        exp_rebal_dates = pd.DatetimeIndex(["2015-01-02", "2015-02-23",
                                            "2015-03-17"])
        assert_index_equal(rebal_dates, exp_rebal_dates)

    def test_tradeables_dates(self):
        exposures = util.make_exposures(
            ["ES", "TY"], self.metadata, self.marketdata
        )
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL
        )
        tradeable_dates = portfolio.tradeable_dates()
        # no CME holdiays between this date range
        exp_tradeable_dates = pd.date_range("2015-01-02", "2015-03-23",
                                            freq="B")
        assert_index_equal(tradeable_dates, exp_tradeable_dates)

        # with an adhoc holiday
        holidays = [pd.Timestamp("2015-01-02", tz="EST")]
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL, holidays=holidays
        )
        tradeable_dates = portfolio.tradeable_dates()
        exp_tradeable_dates = pd.date_range("2015-01-03", "2015-03-23",
                                            freq="B")
        assert_index_equal(tradeable_dates, exp_tradeable_dates)

    def test_simulation_tradeable_futures(self):
        RISK_TARGET = 0.12
        exposures = util.make_exposures(
            ["ES", "TY"], self.metadata, self.marketdata
        )
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL
        )
        signal = util.make_signal(portfolio)

        sim_res = portfolio.simulate(signal, tradeables=True, reinvest=False,
                                     risk_target=RISK_TARGET)

        # since contracts held is only 1 just look at notional of a contract
        es_hlds, es_pnl = util.splice_futures_and_pnl(
          self.marketdata,
          [("ESH2015", "2015-01-02", "2015-03-17"), ("ESM2015", "2015-03-23")]
        )
        # since contracts held is only 1 just look at notional of a contract
        ty_hlds, ty_pnl = util.splice_futures_and_pnl(
          self.marketdata,
          [("TYH2015", "2015-01-02", "2015-02-24"), ("TYM2015", "2015-03-23")]
        )

        hlds_exp = pd.concat([es_hlds, ty_hlds], axis=1, keys=["ES1", "TY1"])
        pnls_exp = pd.concat([es_pnl, ty_pnl], axis=1, keys=["ES1", "TY1"])

        # account for missing settlement pricing data from sources
        trdble_dates = portfolio.tradeable_dates()
        hlds_exp = hlds_exp.reindex(trdble_dates).fillna(method="ffill")
        pnls_exp = pnls_exp.reindex(trdble_dates).fillna(value=0).sum(axis=1)

        # prices for ESH2015, ESH2015, ESM2015, TYH2015, TYH2015, TYM2015
        # for dates 2015-01-02, 2015-03-17, 2015-03-17, 2015-01-02, 2015-02-24, 2015-02-24  # NOQA
        prices = (2046.25, 2074.5, 2066.25, 127.15625, 128.6875, 128.046875)
        mults = (50, 50, 50, 1000, 1000, 1000)
        signals = (1, 1, 1, 1, 1, 1)

        notionals = util.get_notionals(
            RISK_TARGET, self.CAPITAL, signals, prices, mults, discrete=True
        )

        es1 = notionals[0]
        es2 = notionals[2] - notionals[1]
        ty1 = notionals[3]
        ty2 = notionals[5] - notionals[4]

        rebal_dts = portfolio.rebalance_dates()
        trds_exp = pd.DataFrame([[es1, ty1], [0.0, ty2], [es2, 0.0]],
                                index=rebal_dts, columns=["ES1", "TY1"]).round(2)  # NOQA

        exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_simulation_tradeable_futures_multiple_generics(self):
        RISK_TARGET = 0.12
        exposures = util.make_exposures(
            {"ES": ["ES1"], "TY": ["TY1", "TY2"]}, self.metadata,
            self.marketdata
        )
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL
        )
        signal = util.make_signal(portfolio)

        sim_res = portfolio.simulate(signal, tradeables=True, reinvest=False,
                                     risk_target=RISK_TARGET)

        # since contracts held is only 1 just look at notional of a contract
        es_hlds, es_pnl = util.splice_futures_and_pnl(
          self.marketdata,
          [("ESH2015", "2015-01-02", "2015-03-17"), ("ESM2015", "2015-03-23")]
        )
        # since contracts held is only 1 just look at notional of a contract
        ty_hlds, ty_pnl = util.splice_futures_and_pnl(
          self.marketdata,
          [("TYH2015", "2015-01-02", "2015-02-24"), ("TYM2015", "2015-03-23")]
        )
        ty2_hlds, ty2_pnl = util.splice_futures_and_pnl(
          self.marketdata,
          [("TYM2015", "2015-01-02", "2015-02-24"), ("TYU2015", "2015-03-23")]
        )

        hlds_exp = pd.concat([es_hlds, ty_hlds, ty2_hlds],
                             axis=1, keys=["ES1", "TY1", "TY2"])
        pnls_exp = pd.concat([es_pnl, ty_pnl, ty2_pnl],
                             axis=1, keys=["ES1", "TY1", "TY2"])

        # account for missing settlement pricing data from sources
        trdble_dates = portfolio.tradeable_dates()
        hlds_exp = hlds_exp.reindex(trdble_dates).fillna(method="ffill")
        pnls_exp = pnls_exp.reindex(trdble_dates).fillna(value=0).sum(axis=1)

        # prices for ESH2015, ESH2015, ESM2015, TYH2015, TYH2015, TYM2015, TYM2015, TYM2015, TYU2015  # NOQA
        # for dates 2015-01-02, 2015-03-17, 2015-03-17, 2015-01-02, 2015-02-24, 2015-02-24, 2015-01-02, 2015-02-24, 2015-02-24  # NOQA
        prices = (2046.25, 2074.5, 2066.25,
                  127.15625, 128.6875, 128.046875,
                  126.421875, 128.046875, 127.625)
        mults = (50, 50, 50, 1000, 1000, 1000, 1000, 1000, 1000)
        signals = (1, 1, 1, 1, 1, 1, 1, 1, 1)

        notionals = util.get_notionals(
            RISK_TARGET, self.CAPITAL, signals, prices, mults, discrete=True
        )

        es_t1 = notionals[0]
        es_t3 = notionals[2] - notionals[1]
        ty_t1 = notionals[3]
        ty_t2 = notionals[5] - notionals[4]
        ty2_t1 = notionals[6]
        ty2_t2 = notionals[8] - notionals[7]

        rebal_dts = portfolio.rebalance_dates()
        trds_exp = pd.DataFrame([[es_t1, ty_t1, ty2_t1], [0.0, ty_t2, ty2_t2],
                                 [es_t3, 0.0, 0.0]], index=rebal_dts,
                                columns=["ES1", "TY1", 'TY2']).round(2)

        exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_simulation_fungible_futures(self):
        RISK_TARGET = 0.12
        exposures = util.make_exposures(["ES"], self.metadata, self.marketdata)
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL
        )
        sig_val = 1
        signal = util.make_signal(portfolio) * sig_val

        sim_res = portfolio.simulate(signal, tradeables=False, reinvest=False,
                                     risk_target=RISK_TARGET)

        rets = util.splice_returns(
            self.marketdata,
            [("ESH2015", "2015-01-02", "2015-03-17"),
             ("ESM2015", "2015-03-18", "2015-03-23")]
        )
        rets.iloc[0] = 0
        es_hlds1 = (1 + rets.loc[:"2015-03-17"]).cumprod() * self.CAPITAL * sig_val * RISK_TARGET  # NOQA
        rets2 = rets.loc["2015-03-17":]
        rets2.iloc[0] = 0
        es_hlds2 = (1 + rets2.loc["2015-03-17":]).cumprod() * self.CAPITAL * sig_val * RISK_TARGET  # NOQA
        es_hlds = pd.concat([es_hlds1.iloc[:-1], es_hlds2], axis=0)
        es_hlds = pd.DataFrame(es_hlds)
        es_hlds.columns = ["ES1"]

        es_pnl = pd.concat([es_hlds1.diff(), es_hlds2.diff().iloc[1:]], axis=0)
        es_pnl.loc["2015-01-02"] = 0
        # account for missing settlement pricing data from sources
        trdble_dates = portfolio.tradeable_dates()
        hlds_exp = es_hlds.reindex(trdble_dates).fillna(method="ffill")
        pnls_exp = es_pnl.reindex(trdble_dates).fillna(value=0)
        pnls_exp.name = None

        es_pre_rebal_hlds = es_hlds1.loc["2015-03-17"]
        vals = [self.CAPITAL * sig_val * RISK_TARGET,
                self.CAPITAL * sig_val * RISK_TARGET - es_pre_rebal_hlds]
        trds_exp = pd.DataFrame(
            vals,
            index=pd.DatetimeIndex(["2015-01-02", "2015-03-17"]),
            columns=["ES1"]
        )

        exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_simulation_fungible_reinvest_futures(self):
        RISK_TARGET = 0.12
        exposures = util.make_exposures(
            ["ES"], self.metadata, self.marketdata
        )
        portfolio = util.make_portfolio(
            exposures, self.sd, self.ed, self.CAPITAL
        )
        sig_val = 1
        signal = util.make_signal(portfolio) * sig_val

        sim_res = portfolio.simulate(signal, tradeables=False, reinvest=True,
                                     risk_target=RISK_TARGET)

        rets = util.splice_returns(
            self.marketdata,
            [("ESH2015", "2015-01-02", "2015-03-17"),
             ("ESM2015", "2015-03-18", "2015-03-23")]
        )
        rets.iloc[0] = 0
        es_hlds1 = (1 + rets.loc[:"2015-03-17"]).cumprod() * self.CAPITAL * sig_val * RISK_TARGET  # NOQA
        NEW_CAPITAL = es_hlds1.diff().sum() + self.CAPITAL
        rets2 = rets.loc["2015-03-17":]
        rets2.iloc[0] = 0
        es_hlds2 = (1 + rets2.loc["2015-03-17":]).cumprod() * NEW_CAPITAL * sig_val * RISK_TARGET  # NOQA
        es_hlds = pd.concat([es_hlds1.iloc[:-1], es_hlds2], axis=0)
        es_hlds = pd.DataFrame(es_hlds)
        es_hlds.columns = ["ES1"]

        es_pnl = pd.concat([es_hlds1.diff(), es_hlds2.diff().iloc[1:]], axis=0)
        es_pnl.loc["2015-01-02"] = 0
        # account for missing settlement pricing data from sources
        trdble_dates = portfolio.tradeable_dates()
        hlds_exp = es_hlds.reindex(trdble_dates).fillna(method="ffill")
        pnls_exp = es_pnl.reindex(trdble_dates).fillna(value=0)
        pnls_exp.name = None

        es_pre_rebal_hlds = es_hlds1.loc["2015-03-17"]
        vals = [self.CAPITAL * sig_val * RISK_TARGET,
                NEW_CAPITAL * sig_val * RISK_TARGET - es_pre_rebal_hlds]
        trds_exp = pd.DataFrame(
            vals,
            index=pd.DatetimeIndex(["2015-01-02", "2015-03-17"]),
            columns=["ES1"]
        )

        exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_return_calculations(self):
        # https://github.com/pandas-dev/pandas/issues/21200
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2015-01-01", "2015-01-03"), ["A1", "A2"]]
        )
        s = pd.Series([1, 3, 1.5, 1.5, 3, 4.5], index=idx)

        rets = strat.calc_returns(s)
        rets_exp = pd.Series([np.NaN, np.NaN, 0.5, -0.5, 1, 2], index=idx)
        assert_series_equal(rets, rets_exp)
