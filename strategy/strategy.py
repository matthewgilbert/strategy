import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
import os
import mapping as mp
import json
import functools
import collections
import re
import warnings
import pandas_market_calendars as mcal
from abc import ABCMeta, abstractmethod


class Exposures():
    """
    A data container for market data on equities and futures instruments.
    Handles looking up prices as well as expiry information for futures. Also
    stores exchange, multiplier and tradeable name for each instrument.
    """

    _asset_readers = {"future": "read_futures",
                      "equity": "read_equity"}

    _month_map = dict(zip("FGHJKMNQUVXZ", range(1, 13)))

    def __init__(self, prices, expiries, meta_data, security_mapping=None):
        """
        Parameters:
        -----------
        prices: dict of pd.DataFrame
            A dictionary of pd.DataFrames where the key represents the root
            generic name for the instrument (e.g. "ES", "XIV") and the
            DataFrame consists of one of the following two types.
            For equities the DataFrame should contain the column "adj_close"
            and have a DatetimeIndex.
            For futures the DataFrame should contain the column "settle" and
            have a MultiIndex where the top level is a DatetimeIndex and the
            second level consists of instrument names of the form YYYYNNC, e.g.
            "2017ESU".
        expiries: pd.DataFrame
            A pd.DataFrame with columns ["contract", "first_notice",
            "last_trade"] where "first_notice" and "last_trade" must be
            parseable to datetimes with format %Y-%m-%d and "contract" must be
            a string in the form YYYYNNC representing the contract name, e.g.
            "2007ESU".
        meta_data: pd.DataFrame
            A pd.DataFrame of instrument meta data, columns should be names
            of root generics, e.g. ["ES", "TY", "XIV"] and index should contain
            ["exchange", "instrument_type", "multiplier", "generics"] where
            "exchange" is a string of the exchange the instrument is traded on,
            "instrument_type" is {"future", "equity"}, "multiplier" is an
            integer representing the instruments multiplier and "generics" is a
            list of generic names for each future and NaN for equities.
        security_mapping: function
            Callable which returns a market tradeable identifier, if nothing
            given default is identity function.
        """

        if expiries.contract.duplicated().any():
            dupes = expiries.loc[expiries.contract.duplicated(keep=False)]
            raise ValueError("Cannot provide multiple rows for same contract"
                             "\n{0}".format(dupes))

        if sorted(list(prices.keys())) != sorted(list(meta_data.columns)):
            raise ValueError("'meta_data' must contain data on all keys in "
                             "'prices'")

        self._prices = dict([(key, prices[key].copy()) for key in prices])
        exp = expiries.copy()
        exp.loc[:, "year"] = exp.contract.str[:4].apply(lambda x: int(x))
        exp.loc[:, "month"] = exp.contract.str[-1].apply(lambda x:
                                                         self._month_map[x])
        exp.loc[:, "root_generic"] = exp.contract.str[4:6]
        exp.loc[:, "first_notice"] = pd.to_datetime(exp.loc[:, "first_notice"],
                                                    format="%Y-%m-%d")
        exp.loc[:, "last_trade"] = pd.to_datetime(exp.loc[:, "last_trade"],
                                                  format="%Y-%m-%d")
        exp = exp[["contract", "year", "month", "root_generic",
                   "first_notice", "last_trade"]]
        self._expiries = exp

        meta_data = meta_data.copy()
        meta_data.columns.name = "root_generic"
        generic_futures = meta_data.loc["generics", meta_data.loc["instrument_type"] == "future"]  # NOQA
        meta_data = meta_data.drop("generics")
        self._meta_data = meta_data

        future_gnrcs = generic_futures.to_dict()
        roots = []
        generics = []
        for root in future_gnrcs:
            generics.extend(future_gnrcs[root])
            roots.extend(len(future_gnrcs[root]) * [root])

        self._generic_to_root = dict(zip(generics, roots))

        self._future_generics = tuple(sorted(generics))
        self._equities = tuple(meta_data.loc[:, meta_data.loc["instrument_type"] == "equity"].columns)  # NOQA
        self._root_futures = tuple(meta_data.loc[:, meta_data.loc["instrument_type"] == "future"].columns)  # NOQA

        instruments = []
        roots = []
        for root in self.root_futures:
            instrs = self._prices[root].index.levels[1]
            instruments.extend(instrs)
            roots.extend(len(instrs) * [root])

        self._instrument_to_root = dict(zip(instruments, roots))
        self._futures = tuple(instruments)

        if security_mapping is None:
            def security_mapping(x):
                return x

        self._to_tradeable = security_mapping

        sprice, sexp = set(self.futures), set(self.expiries.contract)
        extra_prices = sprice.difference(sexp)
        extra_expiries = sexp.difference(sprice)

        # warn the user if instantiating instance with price and expiry data
        # that do not perfectly coincide
        warning = ""
        if extra_prices:
            warning = (warning + "Futures price data without"
                       " expiry data:{0}\n".format(extra_prices))
        if extra_expiries:
            warning = (warning + "Expiry data without futures"
                       " price data:{0}\n".format(extra_expiries))
        if warning:
            warnings.warn(warning)

    def __repr__(self):
        return (
            "Exposures:\n"
            "----------\n"
            "{0}").format(self.meta_data)

    @property
    def prices(self):
        """
        Dictionary of prices
        """
        return self._prices

    @property
    def expiries(self):
        """
        pd.DataFrame of futures instrument expiry data.
        """
        return self._expiries

    @property
    def meta_data(self):
        """
        pd.DataFrame of instrument meta data
        """
        return self._meta_data

    @property
    def root_futures(self):
        """
        Tuple root generic futures, e.g. ("ES", "TY")
        """
        return self._root_futures

    @property
    def equities(self):
        """
        Tuple of equities, e.g. ("XIV",)
        """
        return self._equities

    @property
    def future_generics(self):
        """
        Tuple of future generics, e.g. ("ES1", "TY1", "TY2")
        """
        return self._future_generics

    @property
    def futures(self):
        """
        Tuple of futures, e.g. ('1997ESU', '1997ESZ')
        """
        return self._futures

    def generic_to_root(self, generics):
        """
        Map generic future to corresponding root future

        Parameters:
        -----------
        generics: list
            list of generic instruments

        Returns:
        --------
        List of corresponding root names
        """
        roots = []
        for gnrc in generics:
            roots.append(self._generic_to_root[gnrc])

        return roots

    def instrument_to_root(self, instruments):
        """
        Map future instrument to corresponding root future

        Parameters:
        -----------
        instruments: list
            list of futures instruments

        Returns:
        --------
        List of corresponding root names
        """
        roots = []
        for instr in instruments:
            roots.append(self._instrument_to_root[instr])

        return roots

    def to_tradeable(self, instrument):
        """
        Map internal instrument name defined in prices to a market tradeable
        name

        Parameters:
        -----------
        instrument: str
            Name of instrument

        Returns:
        --------
        Market tradeable identifier
        """
        return self._to_tradeable(instrument)

    def get_xprices(self, date, root_generics):
        """
        Return pd.Series of instrument prices for a given date for a list of
        root generics

        Parameters:
        -----------
        date: pd.Timestamp
            Desired time for prices
        root_generics: list
            List of strings of root generics to lookup prices for. These
            correspond to the keys in the prices dictionnary used to
            initialize the Exposures.

        Example
        -------
        exposures.get_xprices(pd.Timestamp("2016-01-01"), ["ES", "XIV"])
        """
        futures = [f for f in root_generics if f in self.root_futures]
        equities = [f for f in root_generics if f in self.equities]

        vals = []
        for key in futures:
            s = self._prices[key].loc[(date,)].settle
            vals.append(s)

        for key in equities:
            s = self._prices[key].loc[date]
            s.index = [key]
            vals.append(s)

        prices = pd.concat(vals)
        prices.name = pd.Timestamp(date)
        return prices

    @classmethod
    def from_folder(cls, meta_file, data_folder):
        """
        Initialize and Exposures instance from a meta information file and a
        data folder.

        Parameters:
        -----------
        meta_file: str
            File path name to be parsed by Exposures.parse_meta()
        data_folder: str
            Folder path name to be parsed by Exposures.parse_folder()

        Returns:
        --------
        An Exposures instance.
        """
        meta_data = cls.parse_meta(meta_file)  # NOQA
        prices, expiries = cls.parse_folder(data_folder,
                                            meta_data.loc["instrument_type"])
        return cls(prices, expiries, meta_data)

    @staticmethod
    def parse_meta(meta_file):
        """
        Parse a json file for instrument meta data.

        Parameters:
        -----------
        meta_file: str
            File path of json file. This file should resemble json which can
            be used to instantiate a pd.DataFrame used as the meta_data
            parameter to instantiate the class. An example file is show below.

            {
              "ES": {"exchange": "CME", "instrument_type": "future", "multiplier": 50, "generics":  ["ES1"]},
              "TY": {"exchange": "CME", "instrument_type": "future", "multiplier": 1000, "generics":  ["TY1"]},
              "XIV": {"exchange": "NASDAQ", "instrument_type": "equity", "multiplier": 1}
            }

        Returns:
        --------
        A pd.DataFrame of instrument meta data
        """  # NOQA
        with open(meta_file) as fp:
            meta_data = json.load(fp)
        meta_data = pd.DataFrame(meta_data)
        return meta_data

    @classmethod
    def parse_folder(cls, data_folder, instrument_types):
        """
        Parse market data folder for instrument prices and expiry information.

        Parameters:
        -----------
        data_folder: str
            Folder containing market data files for equities and futures.
            An example of the folder structure is shown below. Folders must be
            consistent with Exposures.read_futures() and
            Exposures.read_equities() respectively.

                marketdata/
                  ES/
                    ESH2017.csv
                    ESM2017.csv
                  TY/
                    ESH2017.csv
                    ESM2017.csv
                  XIV/
                    XIV.csv

        instrument_types: pd.Series
            Series who's index is root generic instrument name and value is the
            instrument type {'future', 'equity'}. The instrument_types.index
            coincides with the folder names which will be loaded.

        Returns:
        --------
        A tuple of a dictionary of prices and a pd.DataFrame of expiries used
        for object instantiation.
        """
        price_fldrs = [os.path.join(data_folder, gn)
                       for gn in instrument_types.index]
        prices = {}
        for gnrc_fldr, ast_info in zip(price_fldrs, instrument_types.items()):
            root_generic, asset_type = ast_info
            asset_type_reader = getattr(cls, cls._asset_readers[asset_type])
            prices[root_generic] = asset_type_reader(gnrc_fldr)

        expiry_file = os.path.join(data_folder, "contract_dates.csv")
        expiries = pd.read_csv(expiry_file)
        futures = instrument_types.loc[instrument_types == "future"].index
        expiries = expiries.loc[expiries.contract.str[4:6].isin(futures)]

        return (prices, expiries)

    @staticmethod
    def read_futures(folder):
        """
        Read futures instrument prices.

        Parameters:
        -----------
        folder: str
            A folder containing csv files of individual futures instruments
            prices. Should contain the column Settle and file names should have
            format [A-Z]{3}[1-9][0-9]{3}.csv, e.g "ESU2007.csv"

        Returns:
        --------
        A pd.DataFrame of price data.
        """
        def name_func(namestr):
            name = os.path.split(namestr)[-1].split('.')[0]
            year = name[-4:]
            code = name[2]
            root_generic = name[:2]
            return year + root_generic + code

        files = [os.path.join(folder, f) for f in os.listdir(folder)]

        bad_files = [f for f in files if not
                     re.match("[A-Z]{3}[1-9][0-9]{3}.csv",
                              os.path.split(f)[-1])]
        if bad_files:
            raise ValueError("The following files are not "
                             "properly named:\n{0}".format(bad_files))

        p = mp.util.read_price_data(files, name_func)
        p = p.loc[:, ["Settle"]]
        p.columns = [col.lower().replace(" ", "_") for col in p.columns]
        return p

    @staticmethod
    def read_equity(folder):
        """
        Read equity instrument prices.

        Parameters:
        -----------
        folder: str
            A folder containing a single csv file of equity prices. File should
            contain the column "Adj Close".

        Returns:
        --------
        A pd.DataFrame of price data.
        """
        files = os.listdir(folder)
        if len(files) > 1:
            raise ValueError("%s should contain only one csv file" % folder)
        file = os.path.join(folder, files[0])
        p = pd.read_csv(file, parse_dates=True, index_col=0)
        p = p.loc[:, ["Adj Close"]]
        p.columns = [col.lower().replace(" ", "_") for col in p.columns]
        return p


class Portfolio(metaclass=ABCMeta):
    """
    A Class to manage simulating and generating trades for a trading strategy
    on futures and equities. The main features include:
        - simulations for trading in notional or discrete instrument size
        - user defined roll rules
        - managing instrument holiday calendars
    """

    def __init__(self, exposures, start_date, end_date,
                 initial_capital=100000, risk_target=0.12):
        """
        Parameters:
        -----------
        exposures: Exposures
            An Exposures instance containing the asset exposures for trading
            and backtesting
        start_date:
            PASS
        end_date:
            PASS
        initial_capital: float
            Starting capital for backtest
        """

        self._exposures = exposures

        calendars = []
        for exchange in self._exposures.meta_data.loc["exchange"].unique():
            calendars.append(mcal.get_calendar(exchange))

        self._start_date = start_date
        self._end_date = end_date
        self._calendars = calendars
        self._capital = initial_capital

    def __repr__(self):
        return (
            "Portfolio Initial Capital: {0}\n"
            "Portfolio Exposures:\n"
            "{1}\n\n"
            "Generic Instruments:{2}\n\n"
            "Date Range:\n"
            "----------\n"
            "Start: {3}\n"
            "End: {4}\n"
        ).format(self._capital, self._exposures, self.generics(),
                 self._start_date, self._end_date)

    def tradeable_dates(self, how='all'):
        """
        Dates which are tradeable based on the calendars for the exchanges that
        the set of instruments in the portfolio trade on.

        Parameters:
        -----------
            how: {'all', 'any'}, default 'all'
               * all: all instruments must be tradeable for date to be included
               * any: any instrument must be tradeable for date to be included

        Returns:
        --------
        DatetimeIndex of tradeable dates
        """

        how = {"all": "inner", "any": "outer"}[how]

        sched = self.schedule(how)
        return sched.index
        # schedule_dates = mcal.date_range(sched, frequency='1D')
        # return schedule_dates

    def schedule(self, how="inner"):
        """
        Merge schedule DataFrames from instrument calendars

        how: {'outer', 'inner'}
            How to merge calendars, see pandas_market_calendars.merge_schedules

        Returns:
        --------
        pandas.DataFrame schedule
        """
        return self._schedule(how, self._start_date, self._end_date)

    @functools.lru_cache()
    def _schedule(self, how, start_date, end_date):
        sched = []
        for cal in self._calendars:
            sched.append(cal.schedule(start_date, end_date))
        return mcal.merge_schedules(sched, how=how)

    def holidays(self):
        """
        pd.CumstomBusinessDay of union of all holidays for underlying calendars
        """
        adhoc_holidays = []
        calendar = AbstractHolidayCalendar()
        for cal in self._calendars:
            adhoc_holidays = np.union1d(adhoc_holidays, cal.adhoc_holidays)
            calendar = AbstractHolidayCalendar(
                rules=calendar.merge(cal.regular_holidays)
            )
        return CustomBusinessDay(
            holidays=adhoc_holidays.tolist(),
            calendar=calendar
        )

    def generics(self):
        """
        Return list of generic exposures, e.g. ["ES1", "ES2", "XIV"]
        """
        return self._exposures.future_generics + self._exposures.equities

    def _split_and_check_generics(self, generics):
        if isinstance(generics, pd.Series):
            idx = generics.index.tolist()
        else:
            idx = generics

        futs = [f for f in idx if f in self._exposures.future_generics]
        eqts = [e for e in idx if e in self._exposures.equities]

        if set(futs + eqts) != set(idx):
            raise ValueError("generics contains unknown values.\n"
                             "Received:\n {0}\n"
                             "Expected in set:\n {1}\n"
                             .format(sorted(idx), sorted(futs + eqts)))

        if isinstance(generics, pd.Series):
            futs = generics.loc[futs]
            eqts = generics.loc[eqts]

        return futs, eqts

    def _split_and_check_instruments(self, instruments):
        idx = instruments.index.tolist()
        futs = [f for f in idx if f in self._exposures.futures]
        eqts = [e for e in idx if e in self._exposures.equities]

        if set(futs + eqts) != set(idx):
            raise ValueError("instruments contains unknown values.\n"
                             "Received:\n {0}\n"
                             "Expected in set:\n {1}\n"
                             .format(sorted(idx), sorted(futs + eqts)))

        return instruments.loc[futs], instruments.loc[eqts]

    @abstractmethod
    def rebalance_dates(self):
        """
        Rebalance days for trading strategy

        Returns
        -------
        pandas.DatetimeIndex
        """
        raise NotImplementedError()

    @abstractmethod
    def instrument_weights(self, dates=None):
        """
        Dictionary of instrument weights for each root generic defining roll
        rules for given dates.

        Parameters:
        -----------
        dates: iterable
            iterable of pandas.Timestamps, if None then self.tradeable_dates()
            is used.

        Returns
        -------
        A dictionary of DataFrames of instrument weights indexed by root
        generic, see mapper.mappings.roller()
        """
        raise NotImplementedError()

    @functools.lru_cache(maxsize=1)
    def continuous_rets(self):
        """
        Calculate asset continuous returns. Continuous futures returns are
        defined by the roll rules in instrument_weights().

        Returns
        -------
        pandas.DataFrame of returns
        """

        irets = {}
        for ast in self._exposures.root_futures:
            irets[ast] = (self._exposures.prices[ast].settle
                          .groupby(level=1).pct_change())

        weights = self.instrument_weights()
        futures_crets = mp.util.calc_rets(irets, weights)

        eprices = []
        equities = self._exposures.equities
        for ast in equities:
            eprices.append(self._exposures.prices[ast].adj_close)

        eprices = pd.concat(eprices, axis=1)
        eprices.columns = equities
        equity_rets = eprices.pct_change()

        crets = pd.concat([futures_crets, equity_rets], axis=1)
        crets = crets.sort_index(axis=1)
        return crets.loc[self.tradeable_dates(), :]

    def simulate(self, signal, tradeables=False, rounder=None,
                 reinvest=True, risk_target=0.12):
        """
        Simulate trading strategy with or without discrete trade sizes and
        revinvested.

        Parameters
        ----------
        signal: pd.DataFrame
            Allocations to generic instruments through time.
        tradeables: boolean
            Calculate trades in notional space or use actual fixed size
            contracts, i.e. discrete trade sizes
        rounder: function
            Function to round pd.Series contracts to integers, if None default
            pd.Series.round is used.
        reinvest: boolean
            Whether to reinvest PnL, i.e. have a capital base that is time
            varying versus running a constant amount of capital and risk
        risk_target: float
            Percentage of capital risk to run

        Returns
        -------
        Tuple of "holdings", "trades" and "pnl" which refer to notional
        holdings, notional trades and portfolio pnl respectively
        """

        if not signal.columns.is_unique:
            raise ValueError("signal must have unique columns")

        rebal_dates = self.rebalance_dates()
        tradeable_dates = self.tradeable_dates()

        required_signal_dts = rebal_dates.intersection(tradeable_dates)
        missing = ~required_signal_dts.isin(signal.index)
        if missing.any():
            raise ValueError("'signal' must contain values for "
                             "dates %s" % required_signal_dts[missing])

        returns = self.continuous_rets()
        capital = self._capital

        current_exp = pd.Series(0, signal.columns)
        trade_lst = []
        trd_dts = []
        notional_exposures = []
        returns = returns.fillna(value=0)
        pnls = []
        crnt_instrs = 0
        weights = self.instrument_weights()
        for dt in tradeable_dates:
            daily_pnl = (current_exp * returns.loc[dt]).sum()
            pnls.append(daily_pnl)
            if reinvest:
                capital += daily_pnl
            current_exp = current_exp * (1 + returns.loc[dt])
            if dt in rebal_dates:
                if tradeables:
                    sig_t = signal.loc[dt].dropna()
                    futs, eqts = self._split_and_check_generics(sig_t.index)
                    rt_futs = self._exposures.generic_to_root(futs)
                    prices_t = self._exposures.get_xprices(dt, rt_futs + eqts)
                    trds = self.trade(dt, crnt_instrs, sig_t, prices_t,
                                      capital, risk_target, rounder, weights)

                    new_exp = self.notional_exposure(dt, crnt_instrs, trds,
                                                     prices_t, weights)
                    # account for fact that trds mapped to notionals might
                    # cover all previous generic holdings
                    new_exp = new_exp.reindex(current_exp.index).fillna(0)

                    # calculate generic notional trades
                    trd_dt = new_exp - current_exp
                    current_exp = new_exp
                else:
                    trd_dt = (signal.loc[dt] * capital * risk_target -
                              current_exp)
                    current_exp = signal.loc[dt] * capital * risk_target

                trade_lst.append(trd_dt)
                trd_dts.append(dt)

            notional_exposures.append(current_exp)
        trades = pd.concat(trade_lst, axis=1, keys=rebal_dates).T
        notional_exposures = pd.concat(notional_exposures, axis=1,
                                       keys=tradeable_dates).T
        pnls = pd.Series(pnls, index=tradeable_dates)

        container = collections.namedtuple("sim_result",
                                           ["holdings", "trades", "pnl"])

        return container(notional_exposures, trades, pnls)

    def trade(self, date, instrument_holdings, unit_risk_exposures, prices,
              capital, risk_target, rounder=None, weights=None):
        """
        Generate instrument trade list.

        Parameters:
        -----------
        date: pandas.Timestamp
            Date for trade
        instrument_holdings: pandas.Series
            Current instrument holdings as integer number of contracts
        unit_risk_exposures: pandas.Series
            Unit risk exposure of desired holdings in generics
        prices: pandas.Series
            Prices for instruments to be traded
        capital: float
            Amount of capital to invest
        risk_target: float
            Percentage of capital risk to run
        rounder: function
            Function to round pd.Series contracts to integers, if None default
            pd.Series.round is used.
        weights: dict
            A dict of DataFrames of instrument weights with a MultiIndex where
            the top level contains pandas.Timestamps and the second level is
            instrument names. The columns consist of generic names. Keys should
            be for different root generics, e.g. 'ES'


        Returns:
        --------
        pandas.Series of instrument trades.
        """

        if rounder is None:
            rounder = pd.Series.round

        if weights is None:
            weights = self.instrument_weights(pd.DatetimeIndex([date]))

        dollar_desired_hlds = capital * risk_target * unit_risk_exposures

        ddh_fut, ddh_eqt = self._split_and_check_generics(dollar_desired_hlds)
        price_fut, price_eqt = self._split_and_check_instruments(prices)
        # to support passing 0 as a proxy to all empty holdings
        if instrument_holdings != 0:
            ih_fut, ih_eqt = self._split_and_check_instruments(
                instrument_holdings
            )
        else:
            ih_fut, ih_eqt = (0, 0)

        eq_trades = rounder(ddh_eqt.divide(price_eqt) - ih_eqt)

        root_futs = self._exposures.generic_to_root(ddh_fut.index)
        weights = dict([(r, weights[r].loc[(date,)]) for r in root_futs])

        root_fut_mults = self._exposures.meta_data.loc["multiplier", root_futs]
        fut_mults = mp.util.get_multiplier(weights, root_fut_mults)

        fut_trds = mp.util.calc_trades(ih_fut, ddh_fut, weights, price_fut,
                                       rounder=rounder, multipliers=fut_mults)
        return pd.concat([eq_trades, fut_trds])

    def notional_exposure(self, date, current_instruments, instrument_trades,
                          prices, weights=None):
        """
        Return generic dollar notional exposures.

        Parameters:
        -----------
        date: pandas.Timestamp
            Date for trade
        current_instruments: pandas.Series
            Current instrument holdings as integer number of contracts
        instrument_trades: pandas.Series
            Instrument trades as integer number of contracts
        prices: pandas.Series
            Prices for instruments to be traded
        weights: dict
            A dict of DataFrames of instrument weights with a MultiIndex where
            the top level contains pandas.Timestamps and the second level is
            instrument names. The columns consist of generic names. Keys should
            be for different root generics, e.g. 'ES'

        Returns:
        --------
        pd.Series of notional exposures for generic instruments.
        """
        if weights is None:
            weights = self.instrument_weights(pd.DatetimeIndex([date]))

        new_instrs = instrument_trades.add(current_instruments, fill_value=0)
        new_instrs = new_instrs[new_instrs != 0]

        new_futs, new_eqts = self._split_and_check_instruments(new_instrs)
        prices_futs, prices_eqts = self._split_and_check_instruments(prices)

        eqts_notional = new_eqts * prices_eqts.loc[new_eqts.index]

        root_futs = self._exposures.instrument_to_root(new_futs.index)
        weights = dict([(r, weights[r].loc[(date,)]) for r in root_futs])
        root_fut_mults = self._exposures.meta_data.loc["multiplier", root_futs]
        multipliers = mp.util.get_multiplier(weights, root_fut_mults)
        futs_notional = mp.util.to_notional(new_futs, prices_futs,
                                            multipliers=multipliers)

        # futs_notional can be empty if all trades rounded down to 0
        # and dropped
        if futs_notional.any():
            gnrc_notionals = mp.mappings.to_generics(futs_notional, weights)
        else:
            gnrc_notionals = pd.Series()

        gnrc_notionals = pd.concat([gnrc_notionals, eqts_notional])

        return gnrc_notionals

    def orders(trades):
        """
        Map to market tradeables on an exchange, e.g 2016TYZ -> ZN Dec 17

        Parameters:
        -----------
        trades: pd.Series
            Series of instrument trades where index is Exposures representation
            of a tradeable instrument

        Returns:
        --------
        pd.Series with index of exchange tradeable names
        """
        trades = trades.copy()
        new_index = []
        for nm in trades.index:
            new_index.append(self.exposures.to_tradeable(nm))

        trades.index = new_index
        return trades
