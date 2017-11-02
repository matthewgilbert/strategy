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
    def __init__(self, offset, *args, **kwargs):
        """
        Parameters:
        -----------
        exposures: Exposures
            An Exposures instance containing the asset exposures for trading
            and backtesting
        start_date:
            TODO
        end_date:
            TODO
        initial_capital: float
            Starting capital for backtest
        get_calendar: function
            Map which takes an exchange name as a string and returns an
            instance of a pandas_market_calendars.MarketCalendar. Default is
            pandas_market_calendars.get_calendar
        holidays: list
            list of timezone aware pd.Timestamps used for holidays in addition
            to exchange holidays associated with instruments
        offset: int
            Number of business days to roll relative to earlier of the
            instruments First Notice and Last Trade date
        """
        super(ExpiryPortfolio, self).__init__(*args, **kwargs)
        self.offset = offset

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
            cntrct_close_by_dates[grp] = dts.loc[:, "close_by_earliest_cntrct"]

        wts = {}
        for root in self._exposures.future_root_and_generics:
            gnrcs = self._exposures.future_root_and_generics[root]
            cols = pd.MultiIndex.from_product([gnrcs, ['front', 'back']])
            idx = [self.offset - 1, self.offset]
            trans = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), len(gnrcs))
            transition = pd.DataFrame(trans, index=idx,
                                      columns=cols)
            wts[root] = mp.mappings.roller(dates,
                                          cntrct_close_by_dates[root],
                                          mp.mappings.static_transition,
                                          transition=transition)
        return wts

    def rebalance_dates(self):
        code_close_by = self._get_close_by_dates()
        holidays = self.holidays().holidays
        rebal_dates = code_close_by.loc[:, "close_by_earliest_cntrct"].sort_values().unique().astype('datetime64[D]')  # NOQA
        rebal_dates = np.busday_offset(rebal_dates, offsets=self.offset,
                                       roll='preceding', holidays=holidays)
        rebal_dates = pd.DatetimeIndex(rebal_dates)
        rebal_dates = rebal_dates[rebal_dates >= self._start_date]
        rebal_dates = rebal_dates[rebal_dates <= self._end_date]
        return rebal_dates

    def _get_close_by_dates(self):
        # calculate the days to rebalance on based on rolling all monthly
        # contracts on the same day based on an offset from the earliest
        # contracts First Notice date
        close_by = self._exposures.expiries.set_index("contract")
        close_by.loc[:, "close_by"] = close_by[["first_notice", "last_trade"]].min(axis=1)  # NOQA
        close_by = close_by.sort_values("close_by")
        # for contracts for each monthly code across all products, take the
        # earliest close by date
        code_close_by = close_by.join(close_by.groupby(["year", "month"]).first(), on=["year", "month"], rsuffix="_earliest_cntrct")  # NOQA
        return code_close_by[["root_generic", "close_by_earliest_cntrct"]]


class DailyRebalancePortfolio(ExpiryPortfolio):

    def rebalance_dates(self):
        return self.tradeable_dates()
