import pandas as pd
import numpy as np
import os
import mapping as mp
import json
import functools
import collections
import re
import warnings

# control data warnings in object instantation, see
# https://docs.python.org/3/library/warnings.html#the-warnings-filter
WARNINGS = "default"


# non vectorized, related to
# https://github.com/pandas-dev/pandas/issues/21200
if pd.__version__.startswith("0.23."):
    def calc_returns(s):
        return s.groupby(level=1).apply(lambda x: x.pct_change())
else:
    def calc_returns(s):
        return s.groupby(level=1).pct_change()


def calc_instr_returns(s, index, limit):
    s = mp.util.reindex(s, index, limit)
    rets = calc_returns(s).reindex(index)
    return rets


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

        if sorted(list(prices.keys())) != sorted(list(meta_data.columns)):
            raise ValueError("'meta_data' must contain data on all keys in "
                             "'prices'")

        self._prices = dict([(key, prices[key].copy()) for key in prices])
        self._expiries = self._validate_expiries(expiries)

        meta_data = meta_data.copy()
        meta_data.columns.name = "root_generic"
        generic_futures = meta_data.loc["generics", meta_data.loc["instrument_type"] == "future"]  # NOQA
        meta_data = meta_data.drop("generics")
        self._meta_data = meta_data

        future_gnrcs = generic_futures.to_dict()
        self._future_root_and_generics = future_gnrcs
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
            with warnings.catch_warnings():
                warnings.simplefilter(WARNINGS)
                warnings.warn(warning)

    @classmethod
    def _validate_expiries(cls, expiries):
        if expiries.contract.duplicated().any():
            dupes = expiries.loc[expiries.contract.duplicated(keep=False)]
            raise ValueError("Cannot provide multiple rows for same contract"
                             "in 'expiries'\n{0}".format(dupes))

        matches = expiries.contract.str.match("[0-9]{4}[a-zA-Z]{2}[FGHJKMNQUVXZ]{1}$")  # NOQA
        if not matches.all():
            raise ValueError("'expiries' contract column must have specified "
                             "format\n{0}".format(expiries.loc[~matches, :]))

        exp = expiries.copy()
        exp.loc[:, "year"] = exp.contract.str[:4].apply(lambda x: int(x))
        exp.loc[:, "month"] = exp.contract.str[-1].apply(lambda x:
                                                         cls._month_map[x])
        exp.loc[:, "root_generic"] = exp.contract.str[4:6]
        exp.loc[:, "first_notice"] = pd.to_datetime(exp.loc[:, "first_notice"],
                                                    format="%Y-%m-%d")
        exp.loc[:, "last_trade"] = pd.to_datetime(exp.loc[:, "last_trade"],
                                                  format="%Y-%m-%d")
        exp = exp[["contract", "year", "month", "root_generic",
                   "first_notice", "last_trade"]]
        return exp

    def __repr__(self):
        return (
            "Exposures:\n"
            "----------\n"
            "{0}\n\n"
            "Generic Futures: {1}\n"
            "Equities: {2}\n").format(self.meta_data, self.future_generics,
                                      self.equities)

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
    def future_root_and_generics(self):
        """
        Dict with key as root generic and value as list of future generics
        """
        return self._future_root_and_generics

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
        root_generics: iterable
            Iterable of strings of root generics to lookup prices for. These
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
    def from_folder(cls, meta_file, data_folder, root_generics=None):
        """
        Initialize and Exposures instance from a meta information file and a
        data folder.

        Parameters:
        -----------
        meta_file: str
            File path name to be parsed by Exposures.parse_meta()
        data_folder: str
            Folder path name to be parsed by Exposures.parse_folder()
        root_generics: list or dict
            If list is given subset of generic instruments to select from the
            instrument meta file. If dict, dict.keys() acts as list for subset
            selection and values should be lists of generics.

        Returns:
        --------
        An Exposures instance.
        """
        meta_data = cls.parse_meta(meta_file)
        if root_generics is not None:
            if isinstance(root_generics, list):
                meta_data = meta_data.loc[:, root_generics]
            else:
                meta_data = meta_data.loc[:, root_generics.keys()]
                for key in root_generics.keys():
                    meta_data.loc["generics", key] = root_generics[key]

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
            An example of the folder structure is shown below. Each instrument
            subfolder must be consistent with Exposures.read_futures() and
            Exposures.read_equities() respectively. contract_dates.csv must be
            readable by Exposures.read_expiries()

                marketdata/
                contract_dates.csv
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
        futures = instrument_types.loc[instrument_types == "future"].index
        expiries = cls.read_expiries(expiry_file, futures)

        return (prices, expiries)

    @staticmethod
    def read_expiries(expiry_file, subset):
        """
        Read expiry information on futures instruments. File should contain
        columns ["contract", "first_notice", "last_trade"] where "first_notice"
        and "last_trade" must be parseable to datetimes with format %Y-%m-%d
        and "contract" must be a string in the form YYYYNNC representing the
        contract name, e.g. "2007ESU".

        Parameters:
        -----------
        expiry_file: str
            csv file to read expiry data.
        subset: set
            Subset of contracts to keep.
        """
        expiries = pd.read_csv(expiry_file)
        expiries = expiries.loc[expiries.contract.str[4:6].isin(subset)]
        return expiries

    @staticmethod
    def read_futures(folder, columns=["Settle"]):
        """
        Read futures instrument prices.

        Parameters:
        -----------
        folder: str
            A folder containing csv files of individual futures instruments
            prices. Should contain the column Settle and file names should have
            format [A-Z]{3}[1-9][0-9]{3}.csv, e.g "ESU2007.csv"
        columns: list
            Columns to return from parsed csv files

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
        p = p.loc[:, columns]
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


def validate_weights_and_rebalances(instrument_weights, rebalance_dates):
    """
    Validate that the rebalance dates are compatible with the instrument
    weights by checking that the set of roll periods implied by the instrument
    weights are contained in the rebalance dates. Raise a ValueError if not
    compatible.

    Parameters
    ----------
    instrument_weights: dictionary
        Dictionary of DataFrames of instrument weights for each root
        generic defining roll rules.
    rebalance_dates: pd.DatetimeIndex
            Dates on which to rebalance
    """

    # relates to Relates to https://github.com/matthewgilbert/strategy/issues/3
    # validate that transitions in instrument weights are in rebal_dates
    for root_generic in instrument_weights:
        wts = instrument_weights[root_generic]
        wts = wts.sort_index().reset_index(level="contract")
        # check if underlying transition matrix is different
        trans = (wts.groupby("date").apply(lambda x: x.values))
        trans_next = trans.shift(-1).ffill()
        changes = ~np.vectorize(np.array_equal)(trans, trans_next)
        instr_dts = wts.index.unique()
        chng_dts = instr_dts[changes]
        invalid_dates = chng_dts.difference(rebalance_dates)
        if not invalid_dates.empty:
            msg = ("instrument_weights['{0}'] has transition on dates "
                   "which are not rebalance dates:\n{1}"
                   .format(root_generic, invalid_dates))
            raise ValueError(msg)


class Portfolio():
    """
    A Class to manage simulating and generating trades for a trading strategy
    on futures and equities. The main features include:
        - simulations for trading in notional or discrete instrument size
        - user defined roll rules
    """

    def __init__(self, exposures, rebalance_dates, mtm_dates,
                 instrument_weights, initial_capital=100000):
        """
        Parameters:
        -----------
        exposures: Exposures
            An Exposures instance containing the asset exposures for trading
            and backtesting
        rebalance_dates: pd.DatetimeIndex
            Dates on which to rebalance
        mtm_dates: pd.DatetimeIndex
            Dates on which to mark to market the portfolio
        instrument_weights: dictionary
            Dictionary of DataFrames of instrument weights for each root
            generic defining roll rules.
        initial_capital: float
            Starting capital for backtest
        """

        self._exposures = exposures
        self._capital = initial_capital

        validate_weights_and_rebalances(instrument_weights, rebalance_dates)
        self._rebalance_dates = rebalance_dates
        self._mtm_dates = mtm_dates
        self._instrument_weights = instrument_weights

    def __repr__(self):
        return (
            "Portfolio Initial Capital: {0}\n"
            "Portfolio Exposures:\n"
            "{1}\n"
            "Date Range:\n"
            "----------\n"
            "Start: {2}\n"
            "End: {3}\n"
        ).format(self._capital, self._exposures,
                 self._rebalance_dates[0], self._rebalance_dates[-1])

    @property
    def equities(self):
        """
        Return tuple of equities defined in portfolio
        """
        return self._exposures.equities

    @property
    def future_generics(self):
        """
        Return tuple of generic futures defined in portfolio
        """
        return self._exposures.future_generics

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

    @property
    def rebalance_dates(self):
        """
        Rebalance days for trading strategy.

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self._rebalance_dates

    @property
    def mtm_dates(self):
        """
        Days for marking to market trading strategy.

        Returns:
        --------
        pandas.DatetimeIndex
        """
        return self._mtm_dates

    @property
    def instrument_weights(self):
        """
        Dictionary of instrument weights for each root generic defining roll
        rules for given dates.

        Returns
        -------
        A dictionary of DataFrames of instrument weights indexed by root
        generic, see mapper.mappings.roller()
        """
        return self._instrument_weights

    def generic_durations(self):
        """
        Return a dictionary with root future generics as keys and
        pandas.DataFrames of future generic durations.

        See also: mapping.util.weighted_expiration()
        """
        wts = self.instrument_weights
        ltd = self._exposures.expiries.set_index("contract").loc[:, "last_trade"]  # NOQA
        durations = {}
        for generic in wts:
            durations[generic] = mp.util.weighted_expiration(wts[generic], ltd)

        return durations

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
        if len(self._exposures.root_futures) > 0:
            weights = self.instrument_weights

        for ast in self._exposures.root_futures:
            widx = weights[ast].index
            irets[ast] = calc_instr_returns(
                self._exposures.prices[ast].settle, widx, limit=0
            )
        if irets:
            futures_crets = mp.util.calc_rets(irets, weights)
        else:
            futures_crets = None

        eprices = []
        equities = self._exposures.equities
        for ast in equities:
            eprices.append(self._exposures.prices[ast].adj_close)

        if eprices:
            eprices = pd.concat(eprices, axis=1)
            eprices.columns = equities
            equity_rets = eprices.pct_change()
        else:
            equity_rets = None

        crets = pd.concat([futures_crets, equity_rets], axis=1)
        crets = crets.sort_index(axis=1)
        return crets.loc[self.mtm_dates, :]

    def simulate(self, signal, tradeables=False, rounder=None,
                 reinvest=True, risk_target=0.12):
        """
        Simulate trading strategy with or without discrete trade sizes and
        revinvested. Holdings are marked to market for each day in
        mtm_dates.

        Parameters
        ----------
        signal: pd.DataFrame
            Allocations to generic instruments through time. Must contain
            values for all rebalance_dates(). This is used as the input to
            trade() or when tradeables=False directly scaled by risk target
            and capital.
        tradeables: boolean
            Calculate trades in notional space or use actual fixed size
            contracts, i.e. discrete trade sizes
        rounder: function
            Function to round pd.Series contracts to integers, if None default
            pd.Series.round is used.
        reinvest: boolean
            Whether PnL is added/subtracted from the capital, i.e. have a
            capital base that is time varying versus running a constant amount
            of capital.
        risk_target: float
            Percentage of capital risk to run, used as input to trade().

        Returns
        -------
        Tuple of "holdings", "trades" and "pnl" which refer to a DataFrame of
        notional holdings, a DataFrame of notional trades and a Series of
        portfolio pnl respectively.
        """

        if not signal.columns.is_unique:
            raise ValueError("signal must have unique columns")

        # validate signal data prior to running a simulation, avoid slow
        # runtime error
        rebal_dates = self.rebalance_dates
        missing = ~rebal_dates.isin(signal.index)
        if missing.any():
            raise ValueError("'signal' must contain values for "
                             "dates %s" % rebal_dates[missing])

        futures, _ = self._split_and_check_generics(signal.columns)
        futures = set(futures)
        # validate pricing data exists for at least one instrument for the
        # dates where there is signal data
        for ast in signal.columns:
            req_price_dts = signal.loc[rebal_dates, ast].dropna().index
            if ast in futures:
                ast = self._exposures.generic_to_root([ast])[0]
            try:
                price_dts = self._exposures.prices[ast].index.levels[0]
            except AttributeError:
                price_dts = self._exposures.prices[ast].index
            isin = req_price_dts.isin(price_dts)
            if not isin.all():
                raise ValueError("Price data in Exposures contained within "
                                 "Portfolio must contain prices for "
                                 "'rebalance_dates' when 'signal' is "
                                 "not NaN, {0} needs prices for:"
                                 "\n{1}\n".format(ast, req_price_dts[~isin]))

        returns = self.continuous_rets()
        capital = self._capital

        current_exp = pd.Series(0, signal.columns)
        trade_lst = []
        trd_dts = []
        notional_exposures = []
        returns = returns.fillna(value=0)
        pnls = []
        crnt_instrs = 0
        tradeable_dates = self.mtm_dates
        for i, dt in enumerate(tradeable_dates):
            # exposure from time dt - 1
            daily_pnl = (current_exp * returns.loc[dt]).sum()
            pnls.append(daily_pnl)
            if reinvest:
                capital += daily_pnl
            # update exposures to time dt
            current_exp = current_exp * (1 + returns.loc[dt])
            if dt in rebal_dates:
                if tradeables:
                    sig_t = signal.loc[dt].dropna()
                    futs, eqts = self._split_and_check_generics(sig_t.index)
                    rt_futs = self._exposures.generic_to_root(futs)
                    # call set() to avoid duplicate rt_futs for cases with
                    # multiple generics, e.g. ES1, ES2
                    prices_t = self._exposures.get_xprices(dt, set(rt_futs + eqts))  # NOQA
                    # this is quite hacky but needed to deal with the fact that
                    # weights on the same day before and after a transition are
                    # different
                    # see https://github.com/matthewgilbert/strategy/issues/1
                    dt_next = tradeable_dates[i + 1]
                    trds = self.trade(dt_next, crnt_instrs, sig_t, prices_t,
                                      capital, risk_target, rounder)

                    new_exp = self.notional_exposure(dt_next, crnt_instrs,
                                                     trds, prices_t)
                    # account for fact that 'trds' mapped to 'new_exp'
                    # (generic notionals) might not span all previous generic
                    # holdings, which should be interpreted as having 0
                    # exposure to this generic now
                    new_exp = new_exp.reindex(current_exp.index).fillna(0)

                    # current_exp and new_exp might differ by epsilon because
                    # current_exp is based on compounded returns vs current
                    # prices
                    trd_ntl = (new_exp - current_exp).round(2)
                    current_exp = new_exp
                    crnt_instrs = trds.add(crnt_instrs, fill_value=0)
                    crnt_instrs = crnt_instrs.loc[crnt_instrs != 0]
                else:
                    trd_ntl = (signal.loc[dt] * capital * risk_target -
                               current_exp)
                    current_exp = signal.loc[dt] * capital * risk_target

                trade_lst.append(trd_ntl)
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
              capital, risk_target, rounder=None):
        """
        Generate instrument trade list.

        Parameters:
        -----------
        date: pandas.Timestamp
            Date for trade
        instrument_holdings: pandas.Series
            Current instrument holdings as integer number of contracts. Can
            pass 0 if there are no instrument holdings.
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

        Returns:
        --------
        pandas.Series of instrument trades.
        """

        if rounder is None:
            rounder = pd.Series.round

        weights = {}
        for root in self.instrument_weights:
            weights[root] = self.instrument_weights[root].loc[[date], :]

        # to support passing 0 as a proxy to all empty holdings
        if isinstance(instrument_holdings, pd.Series):
            if not instrument_holdings.index.is_unique:
                raise ValueError("instrument_holdings must have unique index")
            ih_fut, ih_eqt = self._split_and_check_instruments(
                instrument_holdings
            )
        elif instrument_holdings == 0:
            ih_fut, ih_eqt = (0, 0)
        else:
            raise TypeError("instrument_holdings must be pd.Series or 0")

        dollar_desired_hlds = capital * risk_target * unit_risk_exposures

        ddh_fut, ddh_eqt = self._split_and_check_generics(dollar_desired_hlds)
        price_fut, price_eqt = self._split_and_check_instruments(prices)

        eq_trades = rounder(ddh_eqt.divide(price_eqt) - ih_eqt)

        root_futs = set(self._exposures.generic_to_root(ddh_fut.index))
        weights = dict([(r, weights[r].loc[(date,)]) for r in root_futs])

        root_fut_mults = self._exposures.meta_data.loc["multiplier", root_futs]
        fut_mults = mp.util.get_multiplier(weights, root_fut_mults)

        fut_trds = mp.util.calc_trades(ih_fut, ddh_fut, weights, price_fut,
                                       rounder=rounder, multipliers=fut_mults)
        return pd.concat([eq_trades, fut_trds])

    def notional_exposure(self, date, current_instruments, instrument_trades,
                          prices):
        """
        Return generic dollar notional exposures.

        Parameters:
        -----------
        date: pandas.Timestamp
            Date for trade
        current_instruments: pandas.Series
            Current instrument holdings as integer number of contracts. Can
            pass 0 if all current instrument holdings are 0.
        instrument_trades: pandas.Series
            Instrument trades as integer number of contracts
        prices: pandas.Series
            Prices for instruments to be traded

        Returns:
        --------
        pd.Series of notional exposures for generic instruments.
        """
        weights = {}
        for root in self.instrument_weights:
            weights[root] = self.instrument_weights[root].loc[[date], :]

        if not instrument_trades.index.is_unique:
            raise ValueError('instrument_trades must have unique index')

        if isinstance(current_instruments, pd.Series):
            if not current_instruments.index.is_unique:
                raise ValueError('current_instruments must have unique index')

        new_instrs = instrument_trades.add(current_instruments, fill_value=0)
        new_instrs = new_instrs[new_instrs != 0]

        new_futs, new_eqts = self._split_and_check_instruments(new_instrs)
        prices_futs, prices_eqts = self._split_and_check_instruments(prices)

        eqts_notional = new_eqts * prices_eqts.loc[new_eqts.index]

        root_futs = set(self._exposures.instrument_to_root(new_futs.index))
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

    def orders(self, trades):
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
            new_index.append(self._exposures.to_tradeable(nm))

        trades.index = new_index
        return trades
