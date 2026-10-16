"""Walk-forward forecast of the EIA weekly Lower 48 storage change.

Three models, all OLS refit every week on a trailing three-year window:
  v1       : weekly HDD and CDD, last week's change, five-year average
  weather  : v1 plus squared HDD and CDD (demand is convex in temperature) and
             last week's HDD and CDD, so the model can strip last week's weather
             out of last week's change and keep the structural supply/demand part
  full     : weather plus weekly gas-fired generation (EIA-930), this week and last,
             as the power burn signal
Benchmarks: naive (last week's change) and the five-year average change for the
same ISO week.

Point-in-time rule: the forecast for week t uses degree days and generation through
the Friday that ends week t (both published within a day, before Thursday's EIA
release for that week) and storage data only through week t-1. Generation days that
look like reporting gaps are dropped using a trailing median only.

Usage: python model.py  (reads data/, writes output/)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

TRAIN_WEEKS = 156   # three-year trailing window
GAP_RATIO = 0.5     # a generation day below half its trailing 28-day median is a reporting gap

FEATURES = {
    "v1": ["hdd", "cdd", "change_lag1", "avg5"],
    "weather": ["hdd", "cdd", "hdd2", "cdd2", "hdd_l1", "cdd_l1", "change_lag1", "avg5"],
}
FEATURES["full"] = FEATURES["weather"] + ["pb", "pb_l1"]


def week_ending(dates: pd.Series) -> pd.Series:
    """Label each day with the Friday that ends its EIA week (Saturday to Friday)."""
    return dates + pd.to_timedelta((4 - dates.dt.weekday) % 7, "D")


def clean_power_burn(pb: pd.DataFrame) -> pd.DataFrame:
    """Blank out reporting gaps. A day is a gap when it falls below GAP_RATIO times
    the median of the last 28 days that were themselves accepted, so a gap lasting
    weeks (EIA-930 had one from 2025-11-06 to 2025-12-10) cannot drag the reference
    down with it. Only past days are used."""
    pb = pb.copy().sort_values("date").reset_index(drop=True)
    accepted, keep = [], []
    for v in pb["gas_gen_mwh"].values:
        ok = not (len(accepted) >= 14 and v < GAP_RATIO * np.median(accepted[-28:]))
        keep.append(ok)
        if ok:
            accepted.append(v)
    pb.loc[~np.array(keep), "gas_gen_mwh"] = np.nan
    return pb


def build_features(storage: pd.DataFrame, dd: pd.DataFrame,
                   pb: pd.DataFrame | None = None) -> pd.DataFrame:
    d = dd.copy()
    d["date"] = pd.to_datetime(d["date"])
    if pb is not None:
        p = pb.copy()
        p["date"] = pd.to_datetime(p["date"])
        d = d.merge(clean_power_burn(p), on="date", how="left")
    else:
        d["gas_gen_mwh"] = np.nan
    d["week_ending"] = week_ending(d["date"])
    g = d.groupby("week_ending")
    wk = pd.DataFrame({
        "hdd": g["hdd"].sum(min_count=5),
        "cdd": g["cdd"].sum(min_count=5),
        "pb": g["gas_gen_mwh"].sum(min_count=7) / 1e6,   # TWh per week, all seven days required
    }).reset_index()

    s = storage.copy()
    s["week_ending"] = pd.to_datetime(s["week_ending"])
    df = s.merge(wk, on="week_ending", how="inner").dropna(subset=["change_bcf"])
    df = df.sort_values("week_ending").reset_index(drop=True)
    df["woy"] = df["week_ending"].dt.isocalendar().week.astype(int)
    df["change_lag1"] = df["change_bcf"].shift(1)
    # five-year average change for the same ISO week, from prior years only
    avg5 = []
    for _, row in df.iterrows():
        prior = df[(df["woy"] == row["woy"]) & (df["week_ending"] < row["week_ending"])]
        avg5.append(prior["change_bcf"].tail(5).mean() if len(prior) >= 5 else np.nan)
    df["avg5"] = avg5
    df["hdd2"] = df["hdd"] ** 2 / 100
    df["cdd2"] = df["cdd"] ** 2 / 100
    for c in ["hdd", "cdd", "pb"]:
        df[c + "_l1"] = df[c].shift(1)
    return df


def _fit_predict(train: pd.DataFrame, x: pd.Series, feats) -> float:
    X = np.column_stack([np.ones(len(train)), train[feats].values])
    beta, *_ = np.linalg.lstsq(X, train["change_bcf"].values, rcond=None)
    return float(beta[0] + beta[1:] @ x[feats].values.astype(float))


def walk_forward(df: pd.DataFrame, feats=None) -> pd.DataFrame:
    feats = feats or FEATURES["v1"]
    df = df.dropna(subset=feats + ["change_bcf"]).reset_index(drop=True)
    rows = []
    for t in range(TRAIN_WEEKS, len(df)):
        x = df.iloc[t]
        rows.append({
            "week_ending": x["week_ending"],
            "actual": x["change_bcf"],
            "model": _fit_predict(df.iloc[t - TRAIN_WEEKS:t], x, feats),
            "naive": x["change_lag1"],
            "avg5": x["avg5"],
        })
    return pd.DataFrame(rows)


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """Forecasts from all three models from the first week the full model can run.
    In a week whose generation data has a reporting gap, the full model falls back
    to the weather model's forecast, as a desk would; pb_gap marks those weeks."""
    preds = {name: walk_forward(df, f).set_index("week_ending") for name, f in FEATURES.items()}
    start = preds["full"].index.min()
    out = preds["weather"].loc[start:, ["actual", "naive", "avg5"]].copy()
    for name in FEATURES:
        out[name] = preds[name]["model"]
    out["pb_gap"] = out["full"].isna()
    out["full"] = out["full"].fillna(out["weather"])
    return out.dropna().reset_index()


def score(pred: pd.DataFrame, cols) -> pd.DataFrame:
    out = {}
    for m in cols:
        err = pred["actual"] - pred[m]
        hit = np.sign(pred["actual"] - pred["avg5"]) == np.sign(pred[m] - pred["avg5"])
        out[m] = {
            "MAE_bcf": err.abs().mean(),
            "RMSE_bcf": np.sqrt((err ** 2).mean()),
            "hit_rate_vs_avg5": hit.mean() if m != "avg5" else np.nan,
        }
    return pd.DataFrame(out).T.round(2)


def by_season(pred: pd.DataFrame, cols) -> pd.DataFrame:
    winter = pred["week_ending"].dt.month.isin([11, 12, 1, 2, 3])
    return pd.DataFrame({
        "MAE_Nov_Mar": {m: (pred.loc[winter, "actual"] - pred.loc[winter, m]).abs().mean() for m in cols},
        "MAE_Apr_Oct": {m: (pred.loc[~winter, "actual"] - pred.loc[~winter, m]).abs().mean() for m in cols},
    }).round(2)


CONSENSUS_FILE = "consensus_investing.csv"
BLEND_MIN_WEEKS = 52   # weeks of history before the blend weight is estimated
GAP_THRESHOLDS = [0, 3, 5, 8, 12]


def release_to_week(release: pd.Series) -> pd.Series:
    """EIA releases on Thursday for the week ending the prior Friday; in holiday weeks
    the release moves to Wednesday or Friday. The Friday on or before release - 3 days
    is the report week in all three cases."""
    d = pd.to_datetime(release) - pd.Timedelta(3, "D")
    return d - pd.to_timedelta((d.dt.weekday - 4) % 7, "D")


def load_consensus(cons: pd.DataFrame, storage: pd.DataFrame) -> pd.DataFrame:
    """Keep weeks with a consensus figure, one release per week, and a headline actual
    that agrees with the EIA level change within 1 Bcf (drops data errors and weeks
    where a reclassification makes the two targets differ)."""
    c = cons.copy()
    c["week_ending"] = release_to_week(c["release_date"])
    c = c.dropna(subset=["consensus_bcf"]).drop_duplicates("week_ending", keep=False)
    s = storage[["week_ending", "change_bcf"]].copy()
    s["week_ending"] = pd.to_datetime(s["week_ending"])
    c = c.merge(s, on="week_ending", how="inner")
    c = c[(c["actual_bcf"] - c["change_bcf"]).abs() <= 1]
    return c[["week_ending", "consensus_bcf"]].reset_index(drop=True)


def add_blend(m: pd.DataFrame) -> pd.DataFrame:
    """consensus + w * (model - consensus), with w the OLS slope (through the origin)
    of past surprises on past model-minus-consensus gaps; weeks before
    BLEND_MIN_WEEKS of history get no blend."""
    m = m.sort_values("week_ending").reset_index(drop=True)
    gap = (m["full"] - m["consensus_bcf"]).values
    surprise = (m["actual"] - m["consensus_bcf"]).values
    blend, weight = [], []
    for t in range(len(m)):
        if t < BLEND_MIN_WEEKS:
            blend.append(np.nan)
            weight.append(np.nan)
            continue
        w = (gap[:t] * surprise[:t]).sum() / (gap[:t] ** 2).sum()
        weight.append(w)
        blend.append(m["consensus_bcf"].iloc[t] + w * gap[t])
    m["blend"], m["blend_weight"] = blend, weight
    return m


def _binom_two_sided(k: int, n: int) -> float:
    from math import comb
    tail = sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def consensus_tables(m: pd.DataFrame):
    """Accuracy against consensus, and whether the model's disagreement with consensus
    points the way the surprise goes."""
    acc = {}
    for col in ["full", "weather", "consensus_bcf", "avg5"]:
        e = m["actual"] - m[col]
        acc[col] = {"weeks": len(m), "MAE_bcf": e.abs().mean(), "median_AE_bcf": e.abs().median(),
                    "RMSE_bcf": np.sqrt((e ** 2).mean())}
    b = m.dropna(subset=["blend"])
    for col in ["consensus_bcf", "blend"]:
        e = b["actual"] - b[col]
        acc[col + "_blend_weeks"] = {"weeks": len(b), "MAE_bcf": e.abs().mean(),
                                     "median_AE_bcf": e.abs().median(), "RMSE_bcf": np.sqrt((e ** 2).mean())}
    acc = pd.DataFrame(acc).T.round(2)

    gap = m["full"] - m["consensus_bcf"]
    surprise = m["actual"] - m["consensus_bcf"]
    winter = m["week_ending"].dt.month.isin([11, 12, 1, 2, 3])
    rows = []
    for label, mask in [("all", gap.abs() > 0), ("Nov to Mar", (gap.abs() > 0) & winter),
                        ("Apr to Oct", (gap.abs() > 0) & ~winter)] + \
                       [(f"|gap| > {t} Bcf", gap.abs() > t) for t in GAP_THRESHOLDS[1:]]:
        n = int(mask.sum())
        k = int((np.sign(gap[mask]) == np.sign(surprise[mask])).sum())
        rows.append({"subset": label, "weeks": n, "share_in_model_direction": round(k / n, 3),
                     "p_two_sided": round(_binom_two_sided(k, n), 4)})
    return acc, pd.DataFrame(rows)


def plot(pred: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tail = pred.tail(104)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tail["week_ending"], tail["actual"], color="black", lw=1.6, label="EIA actual")
    ax.plot(tail["week_ending"], tail["full"], color="#1f77b4", lw=1.2, label="Model (weather + power burn)")
    ax.plot(tail["week_ending"], tail["avg5"], color="#999999", lw=1, ls="--", label="5-yr average")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_ylabel("Weekly storage change, Bcf")
    ax.set_title("Lower 48 weekly storage change: walk-forward forecast, last two years")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    storage = pd.read_csv(DATA / "storage.csv")
    dd = pd.read_csv(DATA / "hdd_cdd.csv")
    pb = pd.read_csv(DATA / "power_burn.csv")
    df = build_features(storage, dd, pb)

    pred = run_all(df)
    pred.to_csv(OUT / "predictions.csv", index=False)
    cols = ["full", "weather", "v1", "naive", "avg5"]
    s = score(pred, cols)
    s.to_csv(OUT / "scores.csv")
    season = by_season(pred, cols)
    season.to_csv(OUT / "scores_by_season.csv")

    # weather model alone runs from 2018, before EIA-930 generation data exists
    hist = walk_forward(df, FEATURES["weather"]).rename(columns={"model": "weather"})
    hist_s = score(hist, ["weather", "naive", "avg5"])
    hist_s.to_csv(OUT / "scores_weather_full_history.csv")

    print(f"out-of-sample weeks, all models: {len(pred)}, "
          f"{pred.week_ending.min().date()} to {pred.week_ending.max().date()}")
    print(s.to_string())
    print(season.to_string())
    print(f"\nweather model, longer history: {len(hist)} weeks, "
          f"{hist.week_ending.min().date()} to {hist.week_ending.max().date()}")
    print(hist_s.to_string())
    plot(pred, OUT / "storage_forecast.png")

    if (DATA / CONSENSUS_FILE).exists():
        cons = load_consensus(pd.read_csv(DATA / CONSENSUS_FILE), storage)
        m = add_blend(pred.merge(cons, on="week_ending", how="inner"))
        m.to_csv(OUT / "predictions_vs_consensus.csv", index=False)
        acc, direction = consensus_tables(m)
        acc.to_csv(OUT / "consensus_accuracy.csv")
        direction.to_csv(OUT / "consensus_direction.csv", index=False)
        print(f"\nversus consensus: {len(m)} weeks, "
              f"{m.week_ending.min().date()} to {m.week_ending.max().date()}")
        print(acc.to_string())
        print(direction.to_string(index=False))
