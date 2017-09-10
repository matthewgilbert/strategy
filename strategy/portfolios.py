"""
A collection of useful concrete implementations of a Portfolio classes. Also
serves as a useful reference.
"""

from . import strategy
import pandas as pd
import numpy as np
import mapping as mp
import functools


class QuarterlyPortfolio(strategy.Portfolio):

    def __init__(self, offset, *args, **kwargs):
        super(QuarterlyPortfolio, self).__init__(*args, **kwargs)
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
        for ast in self._exposures.root_futures:
            cols = pd.MultiIndex.from_product([[ast + "1"], ['front', 'back']])
            idx = [self.offset - 1, self.offset]
            transition = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=idx,
                                      columns=cols)
            wts[ast] = mp.mappings.roller(dates,
                                          cntrct_close_by_dates[ast],
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


class DailyRebalancePortfolio(QuarterlyPortfolio):

    def rebalance_dates(self):
        return self.tradeable_dates()
