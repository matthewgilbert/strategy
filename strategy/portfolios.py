"""
A collection of useful concrete implementations of a Portfolio classes. Also
serves as a useful reference.
"""

from . import strategy
import pandas as pd
import numpy as np
import mapping as mp
import functools


class ExpiryPortfolio(strategy.Portfolio):
    __doc__ = (strategy.Portfolio.__doc__ +
               '\nThis is a concrete implementation which rolls each futures '
               'instrument in its\nentirety on a fixed number of days '
               'relative to the earlier of the instruments\nFirst Notice and '
               'Last Trade dates')

    # better to find a DRY method for docstring
    def __init__(self, offset, all_monthly, *args, **kwargs):
        """
        Parameters:
        -----------
        offset: int or dict
            Number of business days to roll relative to earlier of the
            instruments First Notice and Last Trade date. If int is given use
            the same number for all futures, if dict is given keys must cover
            all root generics and contain an integer for each.
        all_monthly: boolean
            Whether to roll each contract individually based on the offset from
            the earlier of its First Notice and Last Trade date or to roll all
            contracts with the same month code based on the earliest date
        exposures: Exposures
            An Exposures instance containing the asset exposures for trading
            and backtesting
        start_date: pandas.Timestamp
            First allowable date when running portfolio simulations
        end_date: pandas.Timestamp
            Last allowable date when running portfolio simulations
        initial_capital: float
            Starting capital for backtest
        get_calendar: function
            Map which takes an exchange name as a string and returns an
            instance of a pandas_market_calendars.MarketCalendar. Default is
            pandas_market_calendars.get_calendar
        holidays: list
            list of timezone aware pd.Timestamps used for holidays in addition
            to exchange holidays associated with instruments
        """
        super(ExpiryPortfolio, self).__init__(*args, **kwargs)
        self._all_monthly = all_monthly

        roots = self._exposures.root_futures
        if not isinstance(offset, dict):
            self._offset = dict(zip(roots, len(roots) * [offset]))
        else:
            self._offset = offset

    def instrument_weights(self, dates=None):
        if dates is None:
            dates = self.tradeable_dates()

        bad_dates = dates.difference(self.tradeable_dates())
        if len(bad_dates) > 0:
            raise ValueError("Dates %s not in calendar" % bad_dates)

        dates = tuple(dates)
        return self._instrument_weights(dates)

    @functools.lru_cache()
    def _instrument_weights(self, dates):
        code_close_by = self._get_close_by_dates()

        cntrct_close_by_dates = {}
        for grp, dts in code_close_by.groupby("root_generic"):
            cntrct_close_by_dates[grp] = dts.loc[:, "close_by"]

        wts = {}
        for root in self._exposures.future_root_and_generics:
            gnrcs = self._exposures.future_root_and_generics[root]
            cols = pd.MultiIndex.from_product([gnrcs, ['front', 'back']])
            offset = self._offset[root]
            idx = [offset, offset + 1]
            trans = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), len(gnrcs))
            transition = pd.DataFrame(trans, index=idx,
                                      columns=cols)
            wts[root] = mp.mappings.roller(dates,
                                           cntrct_close_by_dates[root],
                                           mp.mappings.static_transition,
                                           transition=transition)
        return wts

    def rebalance_dates(self):
        """
        Rebalance days for trading strategy. These are defined as the offset
        given from the earlier of the instruments First Notice and Last Trade
        date. If all_monthly=True then then roll all monthly contracts
        together based on earliest date for this set of contracts. The
        start_date or if this is not a business day the following business day
        is also included in the rebalance days.

        Returns
        -------
        pandas.DatetimeIndex
        """
        close_by = self._get_close_by_dates()
        holidays = self.holidays().holidays
        gnrc_close_by = close_by.groupby(["root_generic"])
        rebal_dates = []
        for gnrc, close_by_dates in gnrc_close_by:
            offset = self._offset[gnrc]
            dates = close_by_dates.loc[:, "close_by"].values.astype('datetime64[D]')  # NOQA
            dates = np.busday_offset(dates, offsets=offset, roll='preceding',
                                     holidays=holidays)
            rebal_dates.append(dates)

        rebal_dates = np.concatenate(rebal_dates)
        rebal_dates = pd.DatetimeIndex(rebal_dates).unique().sort_values()
        rebal_dates = rebal_dates[rebal_dates >= self._start_date]
        rebal_dates = rebal_dates[rebal_dates <= self._end_date]
        first_date = np.busday_offset(self._start_date, 0, roll="following",
                                      holidays=holidays)
        rebal_dates = rebal_dates.union([first_date])

        return rebal_dates

    def _get_close_by_dates(self):
        # calculate the days to offset from for each contract
        close_by = self._exposures.expiries.set_index("contract")
        close_by.loc[:, "close_by"] = close_by[["first_notice", "last_trade"]].min(axis=1)  # NOQA
        close_by = close_by.sort_values("close_by")
        # for contracts for each monthly code across all products, take the
        # earliest close by date
        if self._all_monthly:
            close_by = close_by.join(close_by.groupby(["year", "month"]).first(), on=["year", "month"], rsuffix="_earliest_cntrct")  # NOQA
            close_by = close_by[["root_generic", "close_by_earliest_cntrct"]]
            close_by.columns = ["root_generic", "close_by"]

        return close_by.loc[:, ["root_generic", "close_by"]]


class DailyRebalancePortfolio(ExpiryPortfolio):

    def rebalance_dates(self):
        """
        Rebalance days for trading strategy. These are defined as all tradeable
        dates, i.e. tradeable_dates()

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self.tradeable_dates()


class FixedFrequencyPortfolio(strategy.Portfolio):
    __doc__ = (strategy.Portfolio.__doc__ +
               '\nThis is a concrete implementation for equities. There '
               'are no instrument\nweights and the rebalance is based on a '
               'fixed frequency')

    # better to find a DRY method for docstring
    def __init__(self, frequency, offset, *args, **kwargs):
        """
        Parameters:
        -----------
        frequency: string
            Fixed frequency for reblance, supports {"weekly", "monthly"}
        offset: int or list
            Relative offsets based on the frequency. E.g. [0, 1] for weekly
            gives the first two days of the week, [-5] for monthly gives the
            fifth last day of the month.
        exposures: Exposures
            An Exposures instance containing the asset exposures for trading
            and backtesting
        start_date: pandas.Timestamp
            First allowable date when running portfolio simulations
        end_date: pandas.Timestamp
            Last allowable date when running portfolio simulations
        initial_capital: float
            Starting capital for backtest
        get_calendar: function
            Map which takes an exchange name as a string and returns an
            instance of a pandas_market_calendars.MarketCalendar. Default is
            pandas_market_calendars.get_calendar
        holidays: list
            list of timezone aware pd.Timestamps used for holidays in addition
            to exchange holidays associated with instruments
        """
        if frequency not in ["weekly", "monthly"]:
            raise ValueError("Unknown frequency: {0}".format(frequency))
        super(FixedFrequencyPortfolio, self).__init__(*args, **kwargs)
        self._frequency = frequency
        self._offset = offset

    def rebalance_dates(self):
        holidays = self.holidays()
        if self._frequency == "monthly":
            groups = ["year", "month"]
            sd = self._start_date - pd.offsets.MonthBegin(1)
            ed = self._end_date + pd.offsets.MonthEnd(1)
            dts = pd.date_range(start=sd, end=ed, freq=holidays)
            dts = pd.DataFrame({"date": dts, "month": dts.month,
                               "year": dts.year})
        elif self._frequency == "weekly":
            groups = ["weekofyear"]
            sd = self._start_date - pd.Timedelta(self._start_date.dayofweek,
                                                 unit='D')
            ed = self._end_date + pd.Timedelta(6 - self._end_date.dayofweek,
                                               unit='D')
            dts = pd.date_range(start=sd, end=ed, freq=holidays)
            dts = pd.DataFrame({"date": dts, "weekofyear": dts.weekofyear})

        dts = dts.groupby(groups).apply(lambda x: x.iloc[self._offset])
        dts = pd.DatetimeIndex(dts.loc[:, "date"].reset_index(drop=True))
        dts = dts[(dts > self._start_date) & (dts <= self._end_date)]
        dts = pd.DatetimeIndex([self._start_date]).append(dts)
        return dts

    def instrument_weights(self):
        return {}
