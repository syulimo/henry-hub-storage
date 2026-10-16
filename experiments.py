"""Model selection log: every variant tried against the frozen model, on the same weeks.

Each row changes one thing relative to the frozen full model in model.py and is scored
walk-forward on the same weeks: those from 2022-01-21 where the full model runs (weeks
with an EIA-930 reporting gap excluded). The 260-week window starts later, so it covers
only the weeks it can run, and its 2024 to 2026 column is the comparable one.
Usage: python experiments.py  (writes output/model_selection.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import model

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"
START = "2022-01-21"


WEEKS = None   # weeks the frozen full model scores; every variant is scored on these


def score(p: pd.DataFrame) -> dict:
    p = p[p["week_ending"] >= START]
    if WEEKS is not None:
        p = p[p["week_ending"].isin(WEEKS)]
    e = p["actual"] - p["model"]
    w = p["week_ending"].dt.month.isin([11, 12, 1, 2, 3])
    return {"weeks": len(p), "MAE": e.abs().mean(), "MAE_Nov_Mar": e[w].abs().mean(),
            "MAE_Apr_Oct": e[~w].abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAE_2022_2023": e[p["week_ending"] < "2024"].abs().mean(),
            "MAE_2024_2026": e[p["week_ending"] >= "2024"].abs().mean()}


def weekly_extras(dd: pd.DataFrame, regions: pd.DataFrame | None) -> pd.DataFrame:
    d = dd.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week_ending"] = model.week_ending(d["date"])
    g = d.groupby("week_ending")
    x = pd.DataFrame({"hdd_gas": g["hdd_gas"].sum(min_count=5) if "hdd_gas" in d else np.nan,
                      "hdd_max": g["hdd"].max(),
                      "cold_days": g["hdd"].apply(lambda s: (s >= 35).sum())})
    if regions is not None:
        r = regions.copy()
        r["date"] = pd.to_datetime(r["date"])
        r["week_ending"] = model.week_ending(r["date"])
        gr = r.groupby("week_ending")
        # degree days beyond a daily mean near 35F in producing regions: freeze-off proxy
        x["freeze_tx"] = gr["hdd_div7"].apply(lambda s: np.clip(s - 30, 0, None).sum())
        x["freeze_appalachia"] = gr["hdd_div2"].apply(lambda s: np.clip(s - 40, 0, None).sum())
    return x.reset_index()


def main():
    storage = pd.read_csv(DATA / "storage.csv")
    dd = pd.read_csv(DATA / "hdd_cdd.csv")
    pb = pd.read_csv(DATA / "power_burn.csv")
    rfile = DATA / "hdd_regions.csv"
    regions = pd.read_csv(rfile) if rfile.exists() else None

    df = model.build_features(storage, dd, pb).merge(weekly_extras(dd, regions),
                                                    on="week_ending", how="left")
    df["hdd_gas2"] = df["hdd_gas"] ** 2 / 100
    df["hdd_gas_l1"] = df["hdd_gas"].shift(1)
    if "freeze_tx" in df:
        df["freeze_tx_l1"] = df["freeze_tx"].shift(1)

    full = model.FEATURES["full"]
    swap = lambda f, a, b: [b if c == a else c for c in f]  # noqa: E731
    variants = {
        "frozen full model": full,
        "v1 (linear degree days only)": model.FEATURES["v1"],
        "weather only (no power burn)": model.FEATURES["weather"],
        "without squared degree days": [c for c in full if c not in ("hdd2", "cdd2")],
        "without last week's degree days": [c for c in full if c not in ("hdd_l1", "cdd_l1")],
        "without last week's change": [c for c in full if c != "change_lag1"],
        "power burn without CDD terms": [c for c in full if c not in ("cdd", "cdd2", "cdd_l1")],
        "utility-gas-weighted HDD": swap(swap(swap(full, "hdd", "hdd_gas"), "hdd2", "hdd_gas2"),
                                         "hdd_l1", "hdd_gas_l1"),
        "plus coldest day of the week": full + ["hdd_max"],
        "plus days with HDD of 35 or more": full + ["cold_days"],
    }
    if regions is not None:
        variants.update({
            "plus Texas freeze degree days": full + ["freeze_tx"],
            "plus Texas freeze, this week and last": full + ["freeze_tx", "freeze_tx_l1"],
            "plus Appalachia freeze degree days": full + ["freeze_appalachia"],
            "plus both freeze proxies": full + ["freeze_tx", "freeze_appalachia"],
        })
    global WEEKS
    frozen = model.walk_forward(df, full)
    WEEKS = set(frozen.loc[frozen["week_ending"] >= START, "week_ending"])
    rows = []
    for name, feats in variants.items():
        rows.append({"variant": name, **score(model.walk_forward(df, feats))})

    for weeks in (104, 260):
        saved = model.TRAIN_WEEKS
        model.TRAIN_WEEKS = weeks
        rows.append({"variant": f"training window {weeks} weeks",
                     **score(model.walk_forward(df, full))})
        model.TRAIN_WEEKS = saved

    # Friday-to-Thursday weeks instead of Saturday-to-Friday
    saved = model.week_ending
    model.week_ending = lambda d: d + pd.to_timedelta(
        ((4 - d.dt.weekday) % 7).where(lambda o: o != 0, 7), "D")
    rows.append({"variant": "Friday-to-Thursday weeks",
                 **score(model.walk_forward(model.build_features(storage, dd, pb), full))})
    model.week_ending = saved

    out = pd.DataFrame(rows).round(2)
    out.to_csv(OUT / "model_selection.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
