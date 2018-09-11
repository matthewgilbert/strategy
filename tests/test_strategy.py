import strategy.strategy as strat
import pandas as pd
import numpy as np
import pytest
import os
from pandas.util.testing import assert_frame_equal, assert_series_equal
from . import util

# suppress missing price data warnings in Exposures
strat.WARNINGS = "ignore"


@pytest.fixture
def marketdata():
    return os.path.join(os.path.dirname(__file__), 'marketdata')


@pytest.fixture
def metadata():
    return os.path.join(
        os.path.dirname(__file__), 'marketdata', 'instrument_meta.json'
    )


@pytest.fixture
def capital():
    return 1000000


@pytest.fixture
def sd():
    return pd.Timestamp("2015-01-02")


@pytest.fixture
def ed():
    return pd.Timestamp("2015-03-23")


@pytest.fixture
def offset():
    return -3


def assert_simulation_equal(sim1, sim2):
    assert sim1._fields == sim2._fields
    assert_frame_equal(sim1.holdings, sim2.holdings)
    assert_frame_equal(sim1.trades, sim2.trades)
    assert_series_equal(sim1.pnl, sim2.pnl)


def test_simulation_tradeable_futures(metadata, marketdata, sd, ed,
                                      capital, offset):
    RISK_TARGET = 0.12
    exposures = util.make_exposures(
        ["ES", "TY"], metadata, marketdata
    )
    portfolio = util.make_portfolio(
        exposures, sd, ed, capital, offset
    )
    signal = util.make_signal(portfolio)

    sim_res = portfolio.simulate(signal, tradeables=True, reinvest=False,
                                 risk_target=RISK_TARGET)

    # since contracts held is only 1 just look at notional of a contract
    es_hlds, es_pnl = util.splice_futures_and_pnl(
      marketdata,
      [("ESH2015", "2015-01-02", "2015-03-17"), ("ESM2015", "2015-03-23")]
    )
    # since contracts held is only 1 just look at notional of a contract
    ty_hlds, ty_pnl = util.splice_futures_and_pnl(
      marketdata,
      [("TYH2015", "2015-01-02", "2015-02-24"), ("TYM2015", "2015-03-23")]
    )

    hlds_exp = pd.concat([es_hlds, ty_hlds], axis=1, keys=["ES1", "TY1"])
    pnls_exp = pd.concat([es_pnl, ty_pnl], axis=1, keys=["ES1", "TY1"])

    # account for missing settlement pricing data from sources
    trdble_dates = portfolio.mtm_dates
    hlds_exp = hlds_exp.reindex(trdble_dates).fillna(method="ffill")
    pnls_exp = pnls_exp.reindex(trdble_dates).fillna(value=0).sum(axis=1)

    # prices for ESH2015, ESH2015, ESM2015, TYH2015, TYH2015, TYM2015
    # for dates 2015-01-02, 2015-03-17, 2015-03-17, 2015-01-02, 2015-02-24, 2015-02-24  # NOQA
    prices = (2046.25, 2074.5, 2066.25, 127.15625, 128.6875, 128.046875)
    mults = (50, 50, 50, 1000, 1000, 1000)
    signals = (1, 1, 1, 1, 1, 1)

    notionals = util.get_notionals(
        RISK_TARGET, capital, signals, prices, mults, discrete=True
    )

    es1 = notionals[0]
    es2 = notionals[2] - notionals[1]
    ty1 = notionals[3]
    ty2 = notionals[5] - notionals[4]

    rebal_dts = portfolio.rebalance_dates
    trds_exp = pd.DataFrame([[es1, ty1], [0.0, ty2], [es2, 0.0]],
                            index=rebal_dts, columns=["ES1", "TY1"]).round(2)  # NOQA

    exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
    assert_simulation_equal(sim_res, exp_sim_res)


def test_simulation_tradeable_futures_multiple_generics(
      metadata, marketdata, sd, ed, capital, offset
    ):
    RISK_TARGET = 0.12
    exposures = util.make_exposures(
        {"ES": ["ES1"], "TY": ["TY1", "TY2"]}, metadata, marketdata
    )
    portfolio = util.make_portfolio(
        exposures, sd, ed, capital, offset
    )
    signal = util.make_signal(portfolio)

    sim_res = portfolio.simulate(signal, tradeables=True, reinvest=False,
                                 risk_target=RISK_TARGET)

    # since contracts held is only 1 just look at notional of a contract
    es_hlds, es_pnl = util.splice_futures_and_pnl(
      marketdata,
      [("ESH2015", "2015-01-02", "2015-03-17"), ("ESM2015", "2015-03-23")]
    )
    # since contracts held is only 1 just look at notional of a contract
    ty_hlds, ty_pnl = util.splice_futures_and_pnl(
      marketdata,
      [("TYH2015", "2015-01-02", "2015-02-24"), ("TYM2015", "2015-03-23")]
    )
    ty2_hlds, ty2_pnl = util.splice_futures_and_pnl(
      marketdata,
      [("TYM2015", "2015-01-02", "2015-02-24"), ("TYU2015", "2015-03-23")]
    )

    hlds_exp = pd.concat([es_hlds, ty_hlds, ty2_hlds],
                         axis=1, keys=["ES1", "TY1", "TY2"])
    pnls_exp = pd.concat([es_pnl, ty_pnl, ty2_pnl],
                         axis=1, keys=["ES1", "TY1", "TY2"])

    # account for missing settlement pricing data from sources
    trdble_dates = portfolio.mtm_dates
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
        RISK_TARGET, capital, signals, prices, mults, discrete=True
    )

    es_t1 = notionals[0]
    es_t3 = notionals[2] - notionals[1]
    ty_t1 = notionals[3]
    ty_t2 = notionals[5] - notionals[4]
    ty2_t1 = notionals[6]
    ty2_t2 = notionals[8] - notionals[7]

    rebal_dts = portfolio.rebalance_dates
    trds_exp = pd.DataFrame([[es_t1, ty_t1, ty2_t1], [0.0, ty_t2, ty2_t2],
                             [es_t3, 0.0, 0.0]], index=rebal_dts,
                            columns=["ES1", "TY1", 'TY2']).round(2)

    exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
    assert_simulation_equal(sim_res, exp_sim_res)


def test_simulation_fungible_futures(metadata, marketdata, sd, ed, capital,
                                     offset):
    RISK_TARGET = 0.12
    exposures = util.make_exposures(["ES"], metadata, marketdata)
    portfolio = util.make_portfolio(
        exposures, sd, ed, capital, offset
    )
    sig_val = 1
    signal = util.make_signal(portfolio) * sig_val

    sim_res = portfolio.simulate(signal, tradeables=False, reinvest=False,
                                 risk_target=RISK_TARGET)

    rets = util.splice_returns(
        marketdata,
        [("ESH2015", "2015-01-02", "2015-03-17"),
         ("ESM2015", "2015-03-18", "2015-03-23")]
    )
    rets.iloc[0] = 0
    es_hlds1 = (1 + rets.loc[:"2015-03-17"]).cumprod() * capital * sig_val * RISK_TARGET  # NOQA
    rets2 = rets.loc["2015-03-17":]
    rets2.iloc[0] = 0
    es_hlds2 = (1 + rets2.loc["2015-03-17":]).cumprod() * capital * sig_val * RISK_TARGET  # NOQA
    es_hlds = pd.concat([es_hlds1.iloc[:-1], es_hlds2], axis=0)
    es_hlds = pd.DataFrame(es_hlds)
    es_hlds.columns = ["ES1"]

    es_pnl = pd.concat([es_hlds1.diff(), es_hlds2.diff().iloc[1:]], axis=0)
    es_pnl.loc["2015-01-02"] = 0
    # account for missing settlement pricing data from sources
    trdble_dates = portfolio.mtm_dates
    hlds_exp = es_hlds.reindex(trdble_dates).fillna(method="ffill")
    pnls_exp = es_pnl.reindex(trdble_dates).fillna(value=0)
    pnls_exp.name = None

    es_pre_rebal_hlds = es_hlds1.loc["2015-03-17"]
    vals = [capital * sig_val * RISK_TARGET,
            capital * sig_val * RISK_TARGET - es_pre_rebal_hlds]
    trds_exp = pd.DataFrame(
        vals,
        index=pd.DatetimeIndex(["2015-01-02", "2015-03-17"]),
        columns=["ES1"]
    )

    exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
    assert_simulation_equal(sim_res, exp_sim_res)


def test_simulation_fungible_reinvest_futures(metadata, marketdata, sd, ed,
                                              capital, offset):
    RISK_TARGET = 0.12
    exposures = util.make_exposures(
        ["ES"], metadata, marketdata
    )
    portfolio = util.make_portfolio(
        exposures, sd, ed, capital, offset
    )
    sig_val = 1
    signal = util.make_signal(portfolio) * sig_val

    sim_res = portfolio.simulate(signal, tradeables=False, reinvest=True,
                                 risk_target=RISK_TARGET)

    rets = util.splice_returns(
        marketdata,
        [("ESH2015", "2015-01-02", "2015-03-17"),
         ("ESM2015", "2015-03-18", "2015-03-23")]
    )
    rets.iloc[0] = 0
    es_hlds1 = (1 + rets.loc[:"2015-03-17"]).cumprod() * capital * sig_val * RISK_TARGET  # NOQA
    NEW_CAPITAL = es_hlds1.diff().sum() + capital
    rets2 = rets.loc["2015-03-17":]
    rets2.iloc[0] = 0
    es_hlds2 = (1 + rets2.loc["2015-03-17":]).cumprod() * NEW_CAPITAL * sig_val * RISK_TARGET  # NOQA
    es_hlds = pd.concat([es_hlds1.iloc[:-1], es_hlds2], axis=0)
    es_hlds = pd.DataFrame(es_hlds)
    es_hlds.columns = ["ES1"]

    es_pnl = pd.concat([es_hlds1.diff(), es_hlds2.diff().iloc[1:]], axis=0)
    es_pnl.loc["2015-01-02"] = 0
    # account for missing settlement pricing data from sources
    trdble_dates = portfolio.mtm_dates
    hlds_exp = es_hlds.reindex(trdble_dates).fillna(method="ffill")
    pnls_exp = es_pnl.reindex(trdble_dates).fillna(value=0)
    pnls_exp.name = None

    es_pre_rebal_hlds = es_hlds1.loc["2015-03-17"]
    vals = [capital * sig_val * RISK_TARGET,
            NEW_CAPITAL * sig_val * RISK_TARGET - es_pre_rebal_hlds]
    trds_exp = pd.DataFrame(
        vals,
        index=pd.DatetimeIndex(["2015-01-02", "2015-03-17"]),
        columns=["ES1"]
    )

    exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
    assert_simulation_equal(sim_res, exp_sim_res)


def test_return_calculations():
    # https://github.com/pandas-dev/pandas/issues/21200
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2015-01-01", "2015-01-03"), ["A1", "A2"]]
    )
    s = pd.Series([1, 3, 1.5, 1.5, 3, 4.5], index=idx)

    rets = strat.calc_returns(s)
    rets_exp = pd.Series([np.NaN, np.NaN, 0.5, -0.5, 1, 2], index=idx)
    assert_series_equal(rets, rets_exp)


def test_rebalance_dates_instruments_error():
    # test for https://github.com/matthewgilbert/strategy/issues/3
    wts = {
      "ES": pd.DataFrame(
              [1, 1, 1],
              index=pd.MultiIndex.from_tuples(
                      [(pd.Timestamp("2015-03-13"), "2015ESH"),
                       (pd.Timestamp("2015-03-16"), "2015ESH"),
                       (pd.Timestamp("2015-03-17"), "2015ESM")],
                      names=("date", "contract")
                    ),
              columns=["ES1"]
            )
    }

    # dates include all transitions so no issue
    rebal_dts = pd.DatetimeIndex(
        [pd.Timestamp("2015-03-16")]
    )
    strat.validate_weights_and_rebalances(wts, rebal_dts)

    # missing a weight transition date should raise an error
    rebal_dts = pd.DatetimeIndex(["2015-03-13"])
    with pytest.raises(ValueError):
        strat.validate_weights_and_rebalances(wts, rebal_dts)

    # same as above but special case with no rebalance dates
    rebal_dts = []
    with pytest.raises(ValueError):
        strat.validate_weights_and_rebalances(wts, rebal_dts)

    # no transitions so no op
    wts = {
      "ES": pd.DataFrame(
              [1, 1],
              index=pd.MultiIndex.from_tuples(
                      [(pd.Timestamp("2015-03-13"), "2015ESH"),
                       (pd.Timestamp("2015-03-16"), "2015ESH")],
                      names=("date", "contract")
                    ),
              columns=["ES1"]
            )
    }
    strat.validate_weights_and_rebalances(wts, rebal_dts)

    # multi generics
    wts = {
      "ES": pd.DataFrame(
              [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
              index=pd.MultiIndex.from_tuples(
                      [(pd.Timestamp("2015-03-13"), "2015ESH"),
                       (pd.Timestamp("2015-03-13"), "2015ESM"),
                       (pd.Timestamp("2015-03-16"), "2015ESH"),
                       (pd.Timestamp("2015-03-16"), "2015ESU"),
                       (pd.Timestamp("2015-03-17"), "2015ESM"),
                       (pd.Timestamp("2015-03-17"), "2015ESU")],
                      names=("date", "contract")
                    ),
              columns=["ES1", "ES2"]
            )
    }
    rebal_dts = pd.DatetimeIndex(
        [pd.Timestamp("2015-03-13"), pd.Timestamp("2015-03-16")]
    )
    strat.validate_weights_and_rebalances(wts, rebal_dts)


def test_simulation_fungible_equity(metadata, marketdata, capital):
    RISK_TARGET = 0.12
    exposures = util.make_exposures(
        ["XIV"], metadata, marketdata
    )
    sd = pd.Timestamp("2015-01-02")
    ed = pd.Timestamp("2015-01-13")
    offset = -1
    portfolio = util.make_frequency_portfolio(
        "weekly", offset, exposures, sd, ed, capital=capital
    )
    sig_val = 1
    signal = util.make_signal(portfolio) * sig_val

    sim_res = portfolio.simulate(signal, tradeables=False, reinvest=False,
                                 risk_target=RISK_TARGET)

    fn = os.path.join(marketdata, "XIV", "XIV" + ".csv")
    prices = pd.read_csv(fn, parse_dates=True, index_col=0)
    prices.sort_index(inplace=True)
    prices = prices.loc[:, "Adj Close"]
    rets = prices.loc[sd:ed].pct_change()
    rets.iloc[0] = 0

    hlds_exp1 = (1 + rets.loc[:"2015-01-09"]).cumprod() * capital * sig_val * RISK_TARGET  # NOQA
    rets.loc["2015-01-09"] = 0
    hlds_exp2 = (1 + rets.loc["2015-01-09":]).cumprod() * capital * sig_val * RISK_TARGET  # NOQA
    pnls_exp = pd.concat([hlds_exp1.diff(), hlds_exp2.diff().iloc[1:]],
                         axis=0)
    pnls_exp.loc["2015-01-02"] = 0
    pnls_exp.index.names = [None]
    pnls_exp.name = None

    hlds_exp = pd.concat([hlds_exp1.iloc[:-1], hlds_exp2], axis=0)
    hlds_exp = hlds_exp.to_frame(name="XIV")
    hlds_exp.index.names = [None]

    trd1 = capital * 0.12
    trd2 = capital * 0.12 - hlds_exp1.loc["2015-01-09"]
    trds_exp = pd.DataFrame(
        [trd1, trd2],
        index=pd.DatetimeIndex(["2015-01-02", "2015-01-09"]),
        columns=["XIV"]
    )

    exp_sim_res = util.make_container(hlds_exp, trds_exp, pnls_exp)
    assert_simulation_equal(sim_res, exp_sim_res)
