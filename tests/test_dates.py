from strategy.rebalance import get_relative_to_expiry_rebalance_dates, \
    get_fixed_frequency_rebalance_dates, \
    get_relative_to_expiry_instrument_weights
from strategy.calendar import get_mtm_dates
import pandas as pd
import pytest
from pandas.util.testing import assert_index_equal, assert_frame_equal


def assert_dict_of_frames(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    for key in dict1:
        assert_frame_equal(dict1[key], dict2[key], check_names=False)


def test_tradeables_dates():
    # no CME holdiays between this date range
    sd = pd.Timestamp("2015-01-02")
    ed = pd.Timestamp("2015-03-23")
    exchanges = ["CME"]
    tradeable_dates = get_mtm_dates(sd, ed, exchanges)
    exp_tradeable_dates = pd.date_range(
        "2015-01-02", "2015-03-23", freq="B"
    )
    assert_index_equal(tradeable_dates, exp_tradeable_dates)

    # with an adhoc holiday
    holidays = [pd.Timestamp("2015-01-02")]
    tradeable_dates = get_mtm_dates(sd, ed, exchanges, holidays=holidays)
    exp_tradeable_dates = pd.date_range(
        "2015-01-03", "2015-03-23", freq="B"
    )
    assert_index_equal(tradeable_dates, exp_tradeable_dates)

    # with CME holiday (New Years day)
    sd = pd.Timestamp("2015-01-01")
    ed = pd.Timestamp("2015-01-02")
    tradeable_dates = get_mtm_dates(sd, ed, exchanges)
    exp_tradeable_dates = pd.DatetimeIndex([pd.Timestamp("2015-01-02")])
    assert_index_equal(tradeable_dates, exp_tradeable_dates)


def test_relative_to_expiry_rebalance_dates():
    # each contract rolling individually, same offset
    # change to ES and TY
    sd = pd.Timestamp("2015-01-02")
    ed = pd.Timestamp("2015-03-23")
    expiries = pd.DataFrame(
            [["2015ESH", "2015-03-20", "2015-03-20"],
             ["2015ESM", "2015-06-19", "2015-06-19"],
             ["2015TYH", "2015-02-27", "2015-03-20"],
             ["2015TYM", "2015-05-29", "2015-06-19"]],
            columns=["contract", "first_notice", "last_trade"]
    )
    offsets = -3
    rebal_dates = get_relative_to_expiry_rebalance_dates(
        sd, ed,  expiries, offsets, all_monthly=False, holidays=None
    )
    exp_rebal_dates = pd.DatetimeIndex(
        ["2015-01-02", "2015-02-24", "2015-03-17"]
    )
    assert_index_equal(rebal_dates, exp_rebal_dates)

    # rolling all monthly contracts together, same offset
    rebal_dates = get_relative_to_expiry_rebalance_dates(
        sd, ed,  expiries, offsets, all_monthly=True, holidays=None
    )
    exp_rebal_dates = pd.DatetimeIndex(["2015-01-02", "2015-02-24"])
    assert_index_equal(rebal_dates, exp_rebal_dates)

    # rolling each contract individually, different offset
    offsets = {"ES": -3, "TY": -4}
    rebal_dates = get_relative_to_expiry_rebalance_dates(
        sd, ed,  expiries, offsets, all_monthly=False, holidays=None
    )
    exp_rebal_dates = pd.DatetimeIndex(
        ["2015-01-02", "2015-02-23", "2015-03-17"]
    )
    assert_index_equal(rebal_dates, exp_rebal_dates)


def test_relative_to_expiry_weights():
    expiries = pd.DataFrame(
            [["2015ESH", "2015-03-20", "2015-03-20"],
             ["2015ESM", "2015-06-19", "2015-06-19"],
             ["2015ESU", "2015-09-18", "2015-09-18"],
             ["2015TYH", "2015-03-16", "2015-03-20"],
             ["2015TYM", "2015-05-29", "2015-06-19"],
             ["2015TYU", "2015-08-31", "2015-09-21"]],
            columns=["contract", "first_notice", "last_trade"]
    )
    # one generic and one product
    dts = pd.date_range("2015-03-17", "2015-03-18", freq="B")
    offsets = -3
    root_gnrcs = {"ES": ["ES1"]}
    wts = get_relative_to_expiry_instrument_weights(
        dts, root_gnrcs, expiries, offsets
    )
    exp_wts = {
      "ES": pd.DataFrame(
              [1.0, 1.0],
              index=pd.MultiIndex.from_tuples(
                  [(pd.Timestamp("2015-03-17"), "2015ESH"),
                   (pd.Timestamp("2015-03-18"), "2015ESM")],
                  names=("date", "contract")),
              columns=["ES1"]
            )
    }
    assert_dict_of_frames(wts, exp_wts)

    # multiple products
    dts = pd.date_range("2015-03-13", "2015-03-20", freq="B")
    offsets = -1
    root_gnrcs = {"ES": ["ES1"], "TY": ["TY1"]}
    wts = get_relative_to_expiry_instrument_weights(
        dts, root_gnrcs, expiries, offsets
    )
    exp_wts = {
      "ES": pd.DataFrame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-13"), "2015ESH"),
                             (pd.Timestamp("2015-03-16"), "2015ESH"),
                             (pd.Timestamp("2015-03-17"), "2015ESH"),
                             (pd.Timestamp("2015-03-18"), "2015ESH"),
                             (pd.Timestamp("2015-03-19"), "2015ESH"),
                             (pd.Timestamp("2015-03-20"), "2015ESM"),],
                            names=("date", "contract")),
                         columns=["ES1"]
            ),
      "TY": pd.DataFrame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-13"), "2015TYH"),
                             (pd.Timestamp("2015-03-16"), "2015TYM"),
                             (pd.Timestamp("2015-03-17"), "2015TYM"),
                             (pd.Timestamp("2015-03-18"), "2015TYM"),
                             (pd.Timestamp("2015-03-19"), "2015TYM"),
                             (pd.Timestamp("2015-03-20"), "2015TYM"),],
                            names=("date", "contract")),
                         columns=["TY1"]
            )
    }
    assert_dict_of_frames(wts, exp_wts)

    # multiple generics
    offsets = -1
    dts = pd.date_range("2015-03-19", "2015-03-20", freq="B")
    root_gnrcs = {"ES": ["ES1", "ES2"]}
    wts = get_relative_to_expiry_instrument_weights(
        dts, root_gnrcs, expiries, offsets
    )
    exp_wts = {
      "ES": pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-19"), "2015ESH"),
                             (pd.Timestamp("2015-03-19"), "2015ESM"),
                             (pd.Timestamp("2015-03-20"), "2015ESM"),
                             (pd.Timestamp("2015-03-20"), "2015ESU")],
                            names=("date", "contract")),
                         columns=["ES1", "ES2"]
            )
    }
    assert_dict_of_frames(wts, exp_wts)

    # with dict of offsets
    offsets = {"ES": -4, "TY": -1}
    root_gnrcs = {"ES": ["ES1"], "TY": ["TY1"]}
    dts = pd.date_range("2015-03-13", "2015-03-17", freq="B")
    wts = get_relative_to_expiry_instrument_weights(
        dts, root_gnrcs, expiries, offsets
    )
    exp_wts = {
      "ES": pd.DataFrame([1.0, 1.0, 1.0],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-13"), "2015ESH"),
                             (pd.Timestamp("2015-03-16"), "2015ESH"),
                             (pd.Timestamp("2015-03-17"), "2015ESM")],
                            names=("date", "contract")),
                         columns=["ES1"]
            ),
      "TY": pd.DataFrame([1.0, 1.0, 1.0],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-13"), "2015TYH"),
                             (pd.Timestamp("2015-03-16"), "2015TYM"),
                             (pd.Timestamp("2015-03-17"), "2015TYM")],
                            names=("date", "contract")),
                         columns=["TY1"]
            )
    }
    assert_dict_of_frames(wts, exp_wts)        

    # with holidays for relative roll
    offsets = -1
    root_gnrcs = {"ES": ["ES1"]}
    holidays = [pd.Timestamp("2015-03-19").date()]
    dts = pd.date_range("2015-03-18", "2015-03-19", freq="B")
    wts = get_relative_to_expiry_instrument_weights(
        dts, root_gnrcs, expiries, offsets, holidays=holidays
    )
    exp_wts = {
      "ES": pd.DataFrame([1.0, 1.0],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-18"), "2015ESH"),
                             (pd.Timestamp("2015-03-19"), "2015ESM")],
                            names=("date", "contract")),
                         columns=["ES1"]
            )
    }
    assert_dict_of_frames(wts, exp_wts)

    # with monthly flag
    dts = pd.date_range("2015-03-13", "2015-03-16", freq="B")
    root_gnrcs = {"ES": ["ES1"], "TY": ["TY1"]}
    offsets = -1
    wts = get_relative_to_expiry_instrument_weights(
        dts, root_gnrcs, expiries, offsets, all_monthly=True
    )
    exp_wts = {
      "ES": pd.DataFrame([1.0, 1.0],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-13"), "2015ESH"),
                             (pd.Timestamp("2015-03-16"), "2015ESM")],
                            names=("date", "contract")),
                         columns=["ES1"]
            ),
      "TY": pd.DataFrame([1.0, 1.0],
                         index=pd.MultiIndex.from_tuples(
                            [(pd.Timestamp("2015-03-13"), "2015TYH"),
                             (pd.Timestamp("2015-03-16"), "2015TYM")],
                            names=("date", "contract")),
                         columns=["TY1"]
            )
    }
    assert_dict_of_frames(wts, exp_wts)


def test_fixed_frequency_rebalance_dates():
    sd = pd.Timestamp("2015-01-17")
    ed = pd.Timestamp("2015-01-28")
    freq = "monthly"
    offset = -3
    dts = get_fixed_frequency_rebalance_dates(sd, ed, freq, offset)
    exp_dts = pd.DatetimeIndex(["2015-01-17", "2015-01-28"])
    assert_index_equal(dts, exp_dts)

    ed = pd.Timestamp("2015-02-28")
    dts = get_fixed_frequency_rebalance_dates(sd, ed, freq, offset)
    exp_dts = pd.DatetimeIndex(["2015-01-17", "2015-01-28", "2015-02-25"])
    assert_index_equal(dts, exp_dts)

    offset = [-3, -1]
    ed = pd.Timestamp("2015-01-30")
    dts = get_fixed_frequency_rebalance_dates(sd, ed, freq, offset)
    exp_dts = pd.DatetimeIndex(["2015-01-17", "2015-01-28", "2015-01-30"])
    assert_index_equal(dts, exp_dts)

    sd = pd.Timestamp("2015-01-02")
    ed = pd.Timestamp("2015-01-13")
    freq = "weekly"
    offset = [0, 2]
    dts = get_fixed_frequency_rebalance_dates(sd, ed, freq, offset)
    exp_dts = pd.DatetimeIndex(["2015-01-02", "2015-01-05", "2015-01-07",
                                "2015-01-12"])
    assert_index_equal(dts, exp_dts)
