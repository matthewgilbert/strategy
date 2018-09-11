import pandas as pd
import numpy as np
import mapping as mp
from . import strategy


def get_relative_to_expiry_instrument_weights(dates, root_generics, expiries,
                                              offsets, all_monthly=False,
                                              holidays=None):
    """
    Generate instrument weights for each root generic where the position is
    rolled entirely in one day based on an offset from the earlier of the
    contracts First Notice Date and Last Trade Date.

    Parameters
    ----------
    dates: Iterable
        Iterable of pandas.Timestamps, dates to generate instrument weights
        for.
    root_generics: dict
        Dictionary with key as root generic and value as list of future
        generics, e.g. {"CL": ["CL1", "CL2"]}
    expiries: pd.DataFrame
        A pd.DataFrame with columns ["contract", "first_notice",
        "last_trade"] where "first_notice" and "last_trade" must be
        parseable to datetimes with format %Y-%m-%d and "contract" must be
        a string in the form YYYYNNC representing the contract name, e.g.
        "2007ESU".
    offsets: int or dict
        Number of business days to roll relative to earlier of the
        instruments First Notice and Last Trade date. If int is given use
        the same number for all futures, if dict is given keys must cover
        all root generics and contain an integer for each.
    all_monthly: boolean
        Whether to roll each contract individually based on the offset from
        the earlier of its First Notice and Last Trade date or to roll all
        contracts with the same month code based on the earliest date.
    holidays: list
        list of timezone aware pd.Timestamps used for holidays when calculating
        relative date roll logic.

    Returns
    -------
    A dictionary of DataFrames of instrument weights indexed by root
    generic, see mapper.mappings.roller()

    Examples
    --------
    >>> import strategy.rebalance as rebal
    >>> import pandas as pd
    >>> dts = pd.date_range("2018-01-01", "2018-02-01", freq="B")
    >>> rg = {"CL": ["CL1"], "ES": ["ES1"]}
    >>> exp = pd.DataFrame(
    ...         [["2018CLF", "2018-01-28", "2018-01-27"],
    ...          ["2018CLG", "2018-02-28", "2018-02-27"],
    ...          ["2018ESF", "2018-01-20", "2018-01-21"],
    ...          ["2018ESG", "2018-02-20", "2018-02-21"]],
    ...         columns=["contract", "first_notice", "last_trade"]
    ...       )
    >>> offsets = -5
    >>> rebal.get_relative_to_expiry_instrument_weights(dts, rg, exp, offsets)
    """
    close_by = _close_by_dates(expiries, all_monthly)

    cntrct_close_by_dates = {}
    for grp, dts in close_by.groupby("root_generic"):
        cntrct_close_by_dates[grp] = dts.loc[:, "close_by"]

    wts = {}
    for root in root_generics:
        gnrcs = root_generics[root]
        cols = pd.MultiIndex.from_product([gnrcs, ['front', 'back']])
        if not isinstance(offsets, int):
            offset = offsets[root]
        else:
            offset = offsets
        idx = [offset, offset + 1]
        trans = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), len(gnrcs))
        transition = pd.DataFrame(trans, index=idx,
                                  columns=cols)
        wts[root] = mp.mappings.roller(dates,
                                       cntrct_close_by_dates[root],
                                       mp.mappings.static_transition,
                                       transition=transition,
                                       holidays=holidays)
    return wts


def get_relative_to_expiry_rebalance_dates(start_date, end_date,  expiries,
                                           offsets, all_monthly=False,
                                           holidays=None):
    """
    Rebalance days for trading strategy. These are defined as the offset
    given from the earlier of the instruments First Notice and Last Trade
    date. If all_monthly=True then then roll all monthly contracts
    together based on earliest date for this set of contracts. The
    start_date or if this is not a business day the following business day
    is also included in the rebalance days.

    Parameters
    ----------
    start_date: pandas.Timestamp
            Date to generate rebalance dates starting from
    end_date: pandas.Timestamp
            Date to generate rebalance dates until
    expiries: pd.DataFrame
        A pd.DataFrame with columns ["contract", "first_notice",
        "last_trade"] where "first_notice" and "last_trade" must be
        parseable to datetimes with format %Y-%m-%d and "contract" must be
        a string in the form YYYYNNC representing the contract name, e.g.
        "2007ESU".
    offsets: int or dict
        Number of business days to roll relative to earlier of the
        instruments First Notice and Last Trade date. If int is given use
        the same number for all futures, if dict is given keys must cover
        all root generics and contain an integer for each.
    all_monthly: boolean
        Whether to roll each contract individually based on the offset from
        the earlier of its First Notice and Last Trade date or to roll all
        contracts with the same month code based on the earliest date.
    holidays: list
        list of timezone aware pd.Timestamps used for holidays when calculating
        relative date roll logic.

    Returns
    -------
    pandas.DatetimeIndex

    Examples
    --------
    >>> import strategy.rebalance as rebal
    >>> import pandas as pd
    >>> sd = pd.Timestamp("2018-01-01")
    >>> ed = pd.Timestamp("2018-02-01")
    >>> exp = pd.DataFrame(
    ...         [["2018CLF", "2018-01-28", "2018-01-27"],
    ...          ["2018CLG", "2018-02-28", "2018-02-27"],
    ...          ["2018ESF", "2018-01-20", "2018-01-21"],
    ...          ["2018ESG", "2018-02-20", "2018-02-21"]],
    ...         columns=["contract", "first_notice", "last_trade"]
    ...       )
    >>> offsets = -5
    >>> rebal.get_relative_to_expiry_rebalance_dates(sd, ed, exp, offsets)
    """
    if not holidays:
        holidays = []

    close_by = _close_by_dates(expiries, all_monthly)

    gnrc_close_by = close_by.groupby(["root_generic"])
    rebal_dates = []
    for root, close_by_dates in gnrc_close_by:
        if not isinstance(offsets, int):
            offset = offsets[root]
        else:
            offset = offsets
        dates = (
            close_by_dates.loc[:, "close_by"].values.astype('datetime64[D]')
        )
        dates = np.busday_offset(dates, offsets=offset, roll='preceding',
                                 holidays=holidays)
        rebal_dates.append(dates)

    rebal_dates = np.concatenate(rebal_dates)
    rebal_dates = pd.DatetimeIndex(rebal_dates).unique().sort_values()
    rebal_dates = rebal_dates[rebal_dates >= start_date]
    rebal_dates = rebal_dates[rebal_dates <= end_date]
    first_date = np.busday_offset(start_date.date(), 0,
                                  roll="following", holidays=holidays)
    rebal_dates = rebal_dates.union([first_date])

    return rebal_dates


def _close_by_dates(expiries, all_monthly):
        # hacky, should refactor such that not using private method
        # _validate_expiries
        expiries = strategy.Exposures._validate_expiries(expiries)
        close_by = expiries.set_index("contract")
        close_by.loc[:, "close_by"] = (
            close_by[["first_notice", "last_trade"]].min(axis=1)
        )
        close_by = close_by.sort_values("close_by")
        if all_monthly:
            close_by = (
              close_by.join(close_by.groupby(["year", "month"]).first(),
                            on=["year", "month"], rsuffix="_earliest_cntrct")
            )
            close_by = close_by[["root_generic", "close_by_earliest_cntrct"]]
            close_by.columns = ["root_generic", "close_by"]

        close_by = close_by.loc[:, ["root_generic", "close_by"]]
        return close_by


def get_fixed_frequency_rebalance_dates(start_date, end_date, frequency,
                                        offset):
    """
    Generate reblance dates according to a fixed frequency, e.g. Wednesday of
    every week.

    Parameters
    ----------
    start_date: pandas.Timestamp
            Date to generate rebalance dates starting from
    end_date: pandas.Timestamp
            Date to generate rebalance dates until
    frequency: string
        Fixed frequency for reblance, supports {"weekly", "monthly"}
    offset: int or list
        Relative offsets based on the frequency. E.g. [0, 1] for weekly
        gives the first two days of the week, [-5] for monthly gives the
        fifth last day of the month.

    Returns
    -------
    pandas.DatetimeIndex

    Examples
    --------
    >>> import strategy.rebalance as rebal
    >>> import pandas as pd
    >>> sd = pd.Timestamp("2018-01-01")
    >>> ed = pd.Timestamp("2018-02-01")
    >>> freq = "weekly"
    >>> offsets = 2
    >>> rebal.get_fixed_frequency_rebalance_dates(sd, ed, freq, offsets)
    """
    if frequency == "monthly":
        groups = ["year", "month"]
        sd = start_date - pd.offsets.MonthBegin(1)
        ed = end_date + pd.offsets.MonthEnd(1)
        dts = pd.date_range(start=sd, end=ed, freq="B")
        dts = pd.DataFrame({"date": dts, "month": dts.month,
                           "year": dts.year})
    elif frequency == "weekly":
        groups = ["weekofyear"]
        sd = start_date - pd.Timedelta(start_date.dayofweek, unit='D')
        ed = end_date + pd.Timedelta(6 - end_date.dayofweek, unit='D')
        dts = pd.date_range(start=sd, end=ed, freq="B")
        dts = pd.DataFrame({"date": dts, "weekofyear": dts.weekofyear})

    dts = dts.groupby(groups).apply(lambda x: x.iloc[offset])
    dts = pd.DatetimeIndex(dts.loc[:, "date"].reset_index(drop=True))
    dts = dts[(dts > start_date) & (dts <= end_date)]
    dts = pd.DatetimeIndex([start_date]).append(dts)
    return dts
