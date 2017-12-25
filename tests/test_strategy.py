from strategy.portfolios import ExpiryPortfolio, FixedFrequencyPortfolio
from strategy.strategy import Exposures
import strategy.strategy as strat
import pandas as pd
import unittest
from collections import namedtuple
import os
from pandas.util.testing import assert_frame_equal, assert_series_equal, \
    assert_index_equal

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
        assert_frame_equal(sim1.holdings, sim2.holdings, check_names=False)
        assert_frame_equal(sim1.trades, sim2.trades, check_names=False)
        assert_series_equal(sim1.pnl, sim2.pnl, check_names=False)

    @staticmethod
    def make_container(holdings, trades, pnl):
        container = namedtuple("sim_result", ["holdings", "trades", "pnl"])
        return container(holdings, trades, pnl)

    def make_exposures(self, root_generics):
        return Exposures.from_folder(self.metadata, self.marketdata,
                                     root_generics)

    def make_portfolio(self, exposures, offset=-3, all_monthly=False,
                       **kwargs):

        # allow passing root_generics instead of exposures
        if not isinstance(exposures, Exposures):
            exposures = self.make_exposures(exposures)

        init_capital = self.CAPITAL

        portfolio = ExpiryPortfolio(offset, all_monthly, exposures,
                                    self.sd, self.ed, init_capital, **kwargs)
        return portfolio

    @staticmethod
    def make_signal(portfolio):
        asts = portfolio.future_generics + portfolio.equities
        dates = portfolio.rebalance_dates()
        signal = pd.DataFrame(1, index=dates, columns=asts)
        return signal

    @staticmethod
    def get_notionals(risk_target, capital, signals, prices, multipliers,
                      discrete):
        if discrete:
            def calc(sig, price, mult):
                return round(sig * risk_target * capital / (price * mult)) * price * mult  # NOQA
        else:
            def calc(sig, price, mult):
                return sig * risk_target * capital * price * mult
        notionals = []
        for s_i, p_i, m_i in zip(signals, prices, multipliers):
            notionals.append(calc(s_i, p_i, m_i))
        return notionals

    def read_futures_instr(self, instr):
        fn = os.path.join(self.marketdata, instr[:2], instr + ".csv")
        data = pd.read_csv(fn, parse_dates=True, index_col=0)
        data = data.Settle
        data.sort_index(inplace=True)
        return data

    def splice_futures_and_pnl(self, instr_sd_ed):
        # instr_sd_ed is a list of tuples,
        # e.g. [("ESH2015", sd, ed1), ("ESM2015", ed2)], only sd is given for
        # first contract, assummed consecutive afterwards
        MULTS = {"ES": 50, "TY": 1000}
        prices = []
        pnls = []
        instr, sd, ed = instr_sd_ed[0]
        sd = pd.Timestamp(sd)
        ed = pd.Timestamp(ed)
        price = self.read_futures_instr(instr)
        price = price.loc[sd:ed]
        # drop NaN at start
        pnls.append(price.diff().iloc[1:])
        # since holdings on rebalance day are post rebalance holdings
        prices.append(price.iloc[:-1])

        sd = ed
        for i, instr_ed in enumerate(instr_sd_ed[1:]):
            instr, ed = instr_ed
            ed = pd.Timestamp(ed)
            price = self.read_futures_instr(instr)
            price = price.loc[sd:ed]
            # drop NaN at start
            pnls.append(price.diff().iloc[1:])
            # check for last element
            if i < (len(instr_sd_ed[1:]) - 1):
                prices.append(price.iloc[:-1])
            else:
                prices.append(price)
            sd = ed

        prices = pd.concat(prices, axis=0) * MULTS[instr[:2]]
        pnls = pd.concat(pnls, axis=0) * MULTS[instr[:2]]
        return prices, pnls

    def splice_returns(self, instr_sd_ed):
        # instr_sd_ed is a list of tuples,
        # e.g. [("ESH2015", sd1, ed1), ("ESM2015", sd2, ed2)]

        rets = []
        for instr, sd, ed in instr_sd_ed:
            sd = pd.Timestamp(sd)
            ed = pd.Timestamp(ed)
            price = self.read_futures_instr(instr)
            rets.append(price.pct_change().loc[sd:ed])
        rets = pd.concat(rets, axis=0)
        return rets

    def test_rebalance_dates(self):
        # each contract rolling individually, same offset
        exposures = self.make_exposures(["ES", "TY"])
        portfolio = self.make_portfolio(exposures)
        rebal_dates = portfolio.rebalance_dates()
        exp_rebal_dates = pd.DatetimeIndex(["2015-01-02", "2015-02-24",
                                            "2015-03-17"])
        assert_index_equal(rebal_dates, exp_rebal_dates)

        # rolling all monthly contracts together, same offset
        portfolio = self.make_portfolio(exposures, all_monthly=True)
        rebal_dates = portfolio.rebalance_dates()
        exp_rebal_dates = pd.DatetimeIndex(["2015-01-02", "2015-02-24"])
        assert_index_equal(rebal_dates, exp_rebal_dates)

        # rolling each contract individually, different offset
        portfolio = self.make_portfolio(exposures,
                                        offset={"ES": -3, "TY": -4})
        rebal_dates = portfolio.rebalance_dates()
        exp_rebal_dates = pd.DatetimeIndex(["2015-01-02", "2015-02-23",
                                            "2015-03-17"])
        assert_index_equal(rebal_dates, exp_rebal_dates)

    def test_tradeables_dates(self):
        exposures = self.make_exposures(["ES", "TY"])
        portfolio = self.make_portfolio(exposures)
        tradeable_dates = portfolio.tradeable_dates()
        # no CME holdiays between this date range
        exp_tradeable_dates = pd.date_range("2015-01-02", "2015-03-23",
                                            freq="B")
        assert_index_equal(tradeable_dates, exp_tradeable_dates)

        # with an adhoc holiday
        holidays = [pd.Timestamp("2015-01-02", tz="EST")]
        portfolio = self.make_portfolio(exposures, holidays=holidays)
        tradeable_dates = portfolio.tradeable_dates()
        exp_tradeable_dates = pd.date_range("2015-01-03", "2015-03-23",
                                            freq="B")
        assert_index_equal(tradeable_dates, exp_tradeable_dates)

    def test_simulation_tradeable_futures(self):
        RISK_TARGET = 0.12
        portfolio = self.make_portfolio(["ES", "TY"])
        signal = self.make_signal(portfolio)

        sim_res = portfolio.simulate(signal, tradeables=True, reinvest=False,
                                     risk_target=RISK_TARGET)

        # since contracts held is only 1 just look at notional of a contract
        es_hlds, es_pnl = self.splice_futures_and_pnl([("ESH2015", "2015-01-02", "2015-03-17"),    # NOQA
                                                       ("ESM2015", "2015-03-23")])  # NOQA
        # since contracts held is only 1 just look at notional of a contract
        ty_hlds, ty_pnl = self.splice_futures_and_pnl([("TYH2015", "2015-01-02", "2015-02-24"),  # NOQA
                                                       ("TYM2015", "2015-03-23")])  # NOQA

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

        notionals = self.get_notionals(RISK_TARGET, self.CAPITAL, signals, prices, mults, True)  # NOQA

        es1 = notionals[0]
        es2 = notionals[2] - notionals[1]
        ty1 = notionals[3]
        ty2 = notionals[5] - notionals[4]

        rebal_dts = portfolio.rebalance_dates()
        trds_exp = pd.DataFrame([[es1, ty1], [0.0, ty2], [es2, 0.0]],
                                index=rebal_dts, columns=["ES1", "TY1"]).round(2)  # NOQA

        exp_sim_res = self.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_simulation_tradeable_futures_multiple_generics(self):
        RISK_TARGET = 0.12
        portfolio = self.make_portfolio({"ES": ["ES1"], "TY": ["TY1", "TY2"]})
        signal = self.make_signal(portfolio)

        sim_res = portfolio.simulate(signal, tradeables=True, reinvest=False,
                                     risk_target=RISK_TARGET)

        # since contracts held is only 1 just look at notional of a contract
        es_hlds, es_pnl = self.splice_futures_and_pnl([("ESH2015", "2015-01-02", "2015-03-17"),    # NOQA
                                                       ("ESM2015", "2015-03-23")])  # NOQA
        # since contracts held is only 1 just look at notional of a contract
        ty_hlds, ty_pnl = self.splice_futures_and_pnl([("TYH2015", "2015-01-02", "2015-02-24"),  # NOQA
                                                       ("TYM2015", "2015-03-23")])  # NOQA
        ty2_hlds, ty2_pnl = self.splice_futures_and_pnl([("TYM2015", "2015-01-02", "2015-02-24"),  # NOQA
                                                         ("TYU2015", "2015-03-23")])  # NOQA

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

        notionals = self.get_notionals(RISK_TARGET, self.CAPITAL, signals, prices, mults, True)  # NOQA

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

        exp_sim_res = self.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_simulation_fungible_futures(self):
        RISK_TARGET = 0.12
        portfolio = self.make_portfolio(["ES"])
        sig_val = 1
        signal = self.make_signal(portfolio) * sig_val

        sim_res = portfolio.simulate(signal, tradeables=False, reinvest=False,
                                     risk_target=RISK_TARGET)

        rets = self.splice_returns(
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

        exp_sim_res = self.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_simulation_fungible_reinvest_futures(self):
        RISK_TARGET = 0.12
        portfolio = self.make_portfolio(["ES"])
        sig_val = 1
        signal = self.make_signal(portfolio) * sig_val

        sim_res = portfolio.simulate(signal, tradeables=False, reinvest=True,
                                     risk_target=RISK_TARGET)

        rets = self.splice_returns(
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

        exp_sim_res = self.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)

    def test_rebalance_dates_instruments_error(self):
        # test for https://github.com/matthewgilbert/strategy/issues/3
        portfolio = self.make_portfolio(["ES"])
        rebal_dts = pd.DatetimeIndex([])
        weights = portfolio.instrument_weights()

        def raise_value_error():
            portfolio._validate_weights_and_rebalances(weights, rebal_dts)

        self.assertRaises(ValueError, raise_value_error)

    def test_fixed_frequency_dates(self):
        exposures = self.make_exposures(["XIV"])
        port = FixedFrequencyPortfolio("monthly", -3, exposures=exposures,
                                       start_date="2015-01-17",
                                       end_date="2015-01-28")
        dts = port.rebalance_dates()
        exp_dts = pd.DatetimeIndex(["2015-01-17", "2015-01-28"])
        assert_index_equal(dts, exp_dts)

        port = FixedFrequencyPortfolio("monthly", -3, exposures=exposures,
                                       start_date="2015-01-17",
                                       end_date="2015-02-28")
        dts = port.rebalance_dates()
        exp_dts = pd.DatetimeIndex(["2015-01-17", "2015-01-28", "2015-02-25"])
        assert_index_equal(dts, exp_dts)

        port = FixedFrequencyPortfolio("monthly", [-3, -1],
                                       exposures=exposures,
                                       start_date="2015-01-17",
                                       end_date="2015-01-30")
        dts = port.rebalance_dates()
        exp_dts = pd.DatetimeIndex(["2015-01-17", "2015-01-28", "2015-01-30"])
        assert_index_equal(dts, exp_dts)

        port = FixedFrequencyPortfolio("weekly", [0, 2],
                                       exposures=exposures,
                                       start_date="2015-01-02",
                                       end_date="2015-01-13")

        dts = port.rebalance_dates()
        exp_dts = pd.DatetimeIndex(["2015-01-02", "2015-01-05", "2015-01-07",
                                    "2015-01-12"])
        assert_index_equal(dts, exp_dts)

    def test_simulation_fungible_equity(self):
        RISK_TARGET = 0.12
        exposures = self.make_exposures(["XIV"])
        sd = pd.Timestamp("2015-01-02")
        ed = pd.Timestamp("2015-01-13")
        portfolio = FixedFrequencyPortfolio("weekly", -1,
                                            exposures=exposures,
                                            start_date=sd,
                                            end_date=ed,
                                            initial_capital=self.CAPITAL)
        sig_val = 1
        signal = self.make_signal(portfolio) * sig_val

        sim_res = portfolio.simulate(signal, tradeables=False, reinvest=False,
                                     risk_target=RISK_TARGET)

        fn = os.path.join(self.marketdata, "XIV", "XIV" + ".csv")
        prices = pd.read_csv(fn, parse_dates=True, index_col=0)
        prices.sort_index(inplace=True)
        prices = prices.loc[:, "Adj Close"]
        rets = prices.loc[sd:ed].pct_change()
        rets.iloc[0] = 0

        hlds_exp1 = (1 + rets.loc[:"2015-01-09"]).cumprod() * self.CAPITAL * sig_val * RISK_TARGET  # NOQA
        rets.loc["2015-01-09"] = 0
        hlds_exp2 = (1 + rets.loc["2015-01-09":]).cumprod() * self.CAPITAL * sig_val * RISK_TARGET  # NOQA
        pnls_exp = pd.concat([hlds_exp1.diff(), hlds_exp2.diff().iloc[1:]],
                             axis=0)
        pnls_exp.loc["2015-01-02"] = 0

        hlds_exp = pd.concat([hlds_exp1.iloc[:-1], hlds_exp2], axis=0)
        hlds_exp = hlds_exp.to_frame(name="XIV")

        trd1 = self.CAPITAL * 0.12
        trd2 = self.CAPITAL * 0.12 - hlds_exp1.loc["2015-01-09"]
        trds_exp = pd.DataFrame([trd1, trd2],
                                index=pd.DatetimeIndex(["2015-01-02",
                                                       "2015-01-09"]),
                                columns=["XIV"])

        exp_sim_res = self.make_container(hlds_exp, trds_exp, pnls_exp)
        self.assert_simulation_equal(sim_res, exp_sim_res)
