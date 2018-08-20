import pandas as pd
import os
from collections import namedtuple
from strategy.strategy import Exposures
from strategy.portfolios import ExpiryPortfolio


def make_container(holdings, trades, pnl):
    container = namedtuple("sim_result", ["holdings", "trades", "pnl"])
    return container(holdings, trades, pnl)


def make_exposures(root_generics, meta_fp, market_fp):
    return Exposures.from_folder(meta_fp, market_fp, root_generics)


def make_portfolio(exposures, sd, ed, capital, offset=-3, all_monthly=False,
                   **kwargs):

    portfolio = ExpiryPortfolio(
        offset, all_monthly, exposures, sd, ed, capital, **kwargs
    )
    return portfolio


def make_signal(portfolio):
    asts = portfolio.future_generics + portfolio.equities
    dates = portfolio.rebalance_dates()
    signal = pd.DataFrame(1, index=dates, columns=asts)
    return signal


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


def read_futures_instr(data_path, instr):
    fn = os.path.join(data_path, instr[:2], instr + ".csv")
    data = pd.read_csv(fn, parse_dates=True, index_col=0)
    data = data.Settle
    data.sort_index(inplace=True)
    return data


def splice_futures_and_pnl(data_path, instr_sd_ed):
    # instr_sd_ed is a list of tuples,
    # e.g. [("ESH2015", sd, ed1), ("ESM2015", ed2)], only sd is given for
    # first contract, assummed consecutive afterwards
    MULTS = {"ES": 50, "TY": 1000}
    prices = []
    pnls = []
    instr, sd, ed = instr_sd_ed[0]
    sd = pd.Timestamp(sd)
    ed = pd.Timestamp(ed)
    price = read_futures_instr(data_path, instr)
    price = price.loc[sd:ed]
    # drop NaN at start
    pnls.append(price.diff().iloc[1:])
    # since holdings on rebalance day are post rebalance holdings
    prices.append(price.iloc[:-1])

    sd = ed
    for i, instr_ed in enumerate(instr_sd_ed[1:]):
        instr, ed = instr_ed
        ed = pd.Timestamp(ed)
        price = read_futures_instr(data_path, instr)
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


def splice_returns(data_path, instr_sd_ed):
    # instr_sd_ed is a list of tuples,
    # e.g. [("ESH2015", sd1, ed1), ("ESM2015", sd2, ed2)]
    rets = []
    for instr, sd, ed in instr_sd_ed:
        sd = pd.Timestamp(sd)
        ed = pd.Timestamp(ed)
        price = read_futures_instr(data_path, instr)
        rets.append(price.pct_change().loc[sd:ed])
    rets = pd.concat(rets, axis=0)
    return rets
