import pandas_market_calendars as mcal
import warnings


def get_mtm_dates(start_date, end_date, exchanges, how='all', holidays=None):
    """
    Dates which are tradeable based on the calendars for the exchanges that
    the set of instruments in the portfolio trade on.

    Parameters
    ----------
    start_date: pandas.Timestamp
        Date to generate mtm dates starting from
    end_date: pandas.Timestamp
        Date to generate mtm dates until
    exchanges: list
        list of strings representing exchanges, see pandas_market_calendars
    how: {'all', 'any'}, default 'all'
       * all: all instruments must be tradeable for date to be included
       * any: any instrument must be tradeable for date to be included
    holidays: list
        list of adhoc pd.Timestamps representing additional holidays to
        exclude.

    Returns
    -------
    DatetimeIndex of tradeable dates

    Examples
    --------
    >>> import strategy.calendar as cal
    >>> import pandas as pd
    >>> sd = pd.Timestamp("2018-01-01")
    >>> ed = pd.Timestamp("2018-02-01")
    >>> exchanges = ["CME"]
    >>> holidays = [pd.Timestamp("2018-01-03")]
    >>> cal.get_mtm_dates(sd, ed, exchanges, holidays=holidays)
    """
    if holidays is None:
        holidays = []
    schedules = []
    for exch in exchanges:
        cal = mcal.get_calendar(exch)
        schedules.append(cal.schedule(start_date, end_date))

    how = {"all": "inner", "any": "outer"}[how]
    schedule = mcal.merge_schedules(schedules, how=how)
    mtm_dates = schedule.index
    mtm_dates = mtm_dates.difference(holidays)
    return mtm_dates


def validate_pricing_source(start_date, end_date, exposures):
    """
    Check an exposures object to ensure pricing information is consistent with
    instrument holiday calendar. Raise warnings for missing pricing data

    Parameters
    ----------
    start_date: pandas.Timestamp
        Date to validate exposures pricing from
    end_date: pandas.Timestamp
        Date to validate exposures pricing until
    exposures: strategy.Exposures
        Exposures object to validate pricing info for
    """
    for ast, exch in exposures.meta_data.loc["exchange"].items():
        try:
            ast_dts = exposures.prices[ast].index.levels[0]
        except AttributeError:
            ast_dts = exposures.prices[ast].index
        cal = mcal.get_calendar(exch)
        exch_dts = cal.schedule(start_date, end_date).index
        missing_dts = exch_dts.difference(ast_dts)

        # warn the user if instantiating instance with instruments which
        # don't have price data consistent with given calendars
        warning = ""
        if len(missing_dts) > 0:
            warning = (warning + "Generic instrument {0} on exchange {1} "
                       "missing price data for tradeable calendar dates:\n"
                       "{2}\n".format(ast, exch, missing_dts))
        if warning:
            warnings.warn(warning)
