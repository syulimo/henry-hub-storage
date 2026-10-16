"""Does the model's disagreement with consensus pay in the futures market?

For each EIA release, the return is the NYMEX Henry Hub front-month settlement on
release day over the settlement on the trading day before (log return). That window
holds the release (10:30 ET, settlement 14:30 ET) and also everything else that day.

  Reaction : regress the release-day return on the storage surprise (actual minus
             consensus). A bigger build than expected should push price down.
  Signal   : before the release, hold -sign(model - consensus) of one front-month
             contract from the prior settlement to the release-day settlement. The
             model forecast exists by Monday and the consensus before the release,
             so the position uses nothing published after it is taken.

Roll: NYMEX natural gas stops trading three business days before the delivery month.
If the release falls on that last trading day, the expiring contract can squeeze for
reasons unrelated to storage, so contract 2 is used for both ends. If the prior day is
the last trading day, contract 2 on that day is the contract that is front on release
day, so its price is the starting point.

Usage: python price_reaction.py  (needs output/predictions_vs_consensus.csv from
model.py and data/futures.csv from "python fetch_data.py futures")
"""
from pathlib import Path

import numpy as np
import pandas as pd

import model

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"
COST_PER_ROUND_TRIP = 0.003   # $/MMBtu: three $0.001 ticks for spread, slippage and fees
WEEKS_PER_YEAR = 52


def last_trading_days(dates: pd.Series) -> set:
    """Third-to-last trading day of each calendar month, from the dates that traded."""
    d = pd.Series(pd.to_datetime(dates)).sort_values()
    by_month = d.groupby(d.dt.to_period("M"))
    return {g.iloc[-3] for _, g in by_month if len(g) >= 3}


def release_returns(releases: pd.Series, fut: pd.DataFrame) -> pd.DataFrame:
    fut = fut.dropna(subset=["c1"]).sort_values("date").reset_index(drop=True)
    fut["date"] = pd.to_datetime(fut["date"])
    expiry = last_trading_days(fut["date"])
    by_date = fut.set_index("date")
    rows = []
    for r in pd.to_datetime(releases):
        if r not in by_date.index:
            continue
        prior = fut.loc[fut["date"] < r, "date"]
        if prior.empty:
            continue
        p = prior.iloc[-1]
        if r in expiry:      # release on the front contract's last day: trade contract 2
            contract, start, end = 2, by_date.loc[p, "c2"], by_date.loc[r, "c2"]
        elif p in expiry:    # contract 2 on the prior day is front on release day
            contract, start, end = 2, by_date.loc[p, "c2"], by_date.loc[r, "c1"]
        else:
            contract, start, end = 1, by_date.loc[p, "c1"], by_date.loc[r, "c1"]
        if pd.isna(start) or pd.isna(end):
            continue
        rows.append({"release_date": r, "prior_date": p, "rolled": contract == 2,
                     "start_price": start, "end_price": end, "ret": np.log(end / start)})
    return pd.DataFrame(rows)


def base(pvc: pd.DataFrame, cons_raw: pd.DataFrame) -> pd.DataFrame:
    """Weeks with a model forecast and a consensus, with surprise, gap and position."""
    c = cons_raw.copy()
    c["week_ending"] = model.release_to_week(c["release_date"])
    c["release_date"] = pd.to_datetime(c["release_date"])
    c = c.dropna(subset=["consensus_bcf"]).drop_duplicates("week_ending", keep=False)
    c = c[["week_ending", "release_date"]]
    m = pvc.copy()
    m["week_ending"] = pd.to_datetime(m["week_ending"])
    m = m.merge(c, on="week_ending", how="inner")
    m["surprise"] = m["actual"] - m["consensus_bcf"]
    m["gap"] = m["full"] - m["consensus_bcf"]
    m["position"] = -np.sign(m["gap"])
    return m


def build(pvc: pd.DataFrame, cons_raw: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    m = base(pvc, cons_raw)
    m = m.merge(release_returns(m["release_date"], fut), on="release_date", how="inner")
    m["pnl_gross"] = m["position"] * m["ret"]
    cost = np.where(m["position"] != 0, COST_PER_ROUND_TRIP / m["start_price"], 0.0)
    m["pnl_net"] = m["pnl_gross"] - cost
    m["pnl_perfect_sign"] = -np.sign(m["surprise"]) * m["ret"]
    return m.sort_values("release_date").reset_index(drop=True)


def reaction(m: pd.DataFrame) -> dict:
    x, y = m["surprise"].values, m["ret"].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(resid @ resid / (len(y) - 2) * np.linalg.inv(X.T @ X).diagonal())
    return {"weeks": len(m), "beta_pct_per_10bcf": beta[1] * 1000,
            "t_stat": beta[1] / se[1],
            "r_squared": 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))}


def signal_stats(m: pd.DataFrame, label: str) -> dict:
    s = m[m["position"] != 0]
    out = {"subset": label, "weeks": len(s)}
    for col in ["pnl_gross", "pnl_net"]:
        r = s[col]
        out[col + "_mean_bp"] = r.mean() * 1e4
        out[col + "_t"] = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        out[col + "_sharpe_ann"] = r.mean() / r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    out["hit_rate"] = (s["pnl_gross"] > 0).mean()
    out["perfect_sign_mean_bp"] = s["pnl_perfect_sign"].mean() * 1e4
    return out


def summary(m: pd.DataFrame) -> pd.DataFrame:
    winter = m["week_ending"].dt.month.isin([11, 12, 1, 2, 3])
    rows = [signal_stats(m, "all"),
            signal_stats(m[winter], "Nov to Mar weeks"),
            signal_stats(m[~winter], "Apr to Oct weeks"),
            signal_stats(m[m["gap"].abs() > 5], "|gap| > 5 Bcf")]
    return pd.DataFrame(rows).round(3)


def intraday_returns(hourly: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    """Front-month return around the release from Yahoo NG=F hourly bars (US Eastern):
    10:00 to 11:00 for Thursday releases (10:30 ET), 10:00 to 13:00 for Wednesday or
    Friday holiday releases, which come out later in the day. Release days on or one
    day before a last trading day are dropped, since NG=F may switch contracts."""
    h = hourly.copy()
    h["release_date"] = pd.to_datetime(h["release_date"])
    ltd = sorted(last_trading_days(trading_dates))
    td = pd.Series(sorted(pd.to_datetime(trading_dates)))
    near = set(ltd)
    for d in ltd:
        before = td[td < d]
        if len(before):
            near.add(before.iloc[-1])
    h = h[~h["release_date"].isin(near)]
    thursday = h["release_date"].dt.weekday == 3
    end = np.where(thursday, h["close_1100"], h["close_1300"])
    h["ret_intraday"] = np.log(end / h["open_1000"])
    h["window"] = np.where(thursday, "10:00-11:00", "10:00-13:00")
    return h[["release_date", "open_1000", "ret_intraday", "window"]].dropna()


def intraday_stats(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x, y = m["surprise"].values, m["ret_intraday"].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(resid @ resid / (len(y) - 2) * np.linalg.inv(X.T @ X).diagonal())
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    winter = m["week_ending"].dt.month.isin([11, 12, 1, 2, 3])
    for label, s in [("all", m), ("Nov to Mar weeks", m[winter]), ("Apr to Oct weeks", m[~winter]),
                     ("|gap| > 5 Bcf", m[m["gap"].abs() > 5])]:
        s = s[s["position"] != 0]
        g = s["position"] * s["ret_intraday"]
        n = g - COST_PER_ROUND_TRIP / s["open_1000"]
        rows.append({"subset": label, "weeks": len(s),
                     "gross_mean_bp": g.mean() * 1e4, "gross_t": g.mean() / (g.std(ddof=1) / np.sqrt(len(g))),
                     "net_mean_bp": n.mean() * 1e4, "net_t": n.mean() / (n.std(ddof=1) / np.sqrt(len(n))),
                     "hit_rate": (g > 0).mean(),
                     "perfect_sign_mean_bp": (-np.sign(s["surprise"]) * s["ret_intraday"]).mean() * 1e4})
    out = pd.DataFrame(rows)
    out.insert(1, "reaction_beta_pct_per_10bcf", [beta[1] * 1000] + [np.nan] * 3)
    out.insert(2, "reaction_t", [beta[1] / se[1]] + [np.nan] * 3)
    out.insert(3, "reaction_r2", [r2] + [np.nan] * 3)
    return out.round(3)


def load_futures(eia: pd.DataFrame, yahoo: pd.DataFrame | None) -> pd.DataFrame:
    """EIA publishes contracts 1 and 2 only through 2024-04-05. After that, contract 1
    comes from Yahoo Finance NG=F, which matched EIA contract 1 to the tenth of a cent
    on all 567 overlapping days (2022-01-03 to 2024-04-05). Contract 2 is missing after
    2024-04-05, so release weeks whose prior day is a last trading day are dropped."""
    eia = eia.copy()
    eia["date"] = pd.to_datetime(eia["date"])
    if yahoo is None:
        return eia
    y = yahoo.copy()
    y["date"] = pd.to_datetime(y["date"])
    y = y[y["date"] > eia["date"].max()]
    return pd.concat([eia, y.assign(c2=np.nan)], ignore_index=True).sort_values("date")


if __name__ == "__main__":
    pvc = pd.read_csv(OUT / "predictions_vs_consensus.csv")
    cons = pd.read_csv(DATA / model.CONSENSUS_FILE)
    yfile = DATA / "futures_yahoo.csv"
    fut = load_futures(pd.read_csv(DATA / "futures.csv"),
                       pd.read_csv(yfile) if yfile.exists() else None)
    m = build(pvc, cons, fut)
    expected = len(pvc)
    print(f"weeks with consensus: {expected}; dropped for a roll with no contract 2 "
          f"price or missing prices: {expected - len(m)}")
    m.to_csv(OUT / "price_reaction.csv", index=False)
    rx = pd.DataFrame([reaction(m)]).round(3)
    rx.to_csv(OUT / "price_reaction_beta.csv", index=False)
    sm = summary(m)
    sm.to_csv(OUT / "price_reaction_signal.csv", index=False)
    print(f"release weeks with prices: {len(m)}, {m.release_date.min().date()} to "
          f"{m.release_date.max().date()}, rolls handled: {int(m.rolled.sum())}")
    print("\nprice reaction to the surprise (actual minus consensus):")
    print(rx.to_string(index=False))
    print("\nholding -sign(model minus consensus) from prior settlement to release-day settlement:")
    print(sm.to_string(index=False))

    hfile = DATA / "futures_hourly_yahoo.csv"
    if hfile.exists():
        intra = intraday_returns(pd.read_csv(hfile), fut["date"])
        mi = base(pvc, cons).merge(intra, on="release_date", how="inner")
        mi.to_csv(OUT / "price_reaction_intraday.csv", index=False)
        st = intraday_stats(mi)
        st.to_csv(OUT / "price_reaction_intraday_signal.csv", index=False)
        print(f"\nintraday window around the release: {len(mi)} weeks, "
              f"{mi.release_date.min().date()} to {mi.release_date.max().date()}")
        print(st.to_string(index=False))
