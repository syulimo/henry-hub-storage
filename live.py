"""Frozen live record: forecast each week before the EIA release, score it after.

  python live.py forecast   log the forecast for the first unreleased week, once its
                            degree days and generation are complete (all seven days)
  python live.py score      fill in the EIA actual and the consensus for logged weeks

The log is append-only: a week already in forecasts/live_log.csv is never rewritten,
and each row carries the SHA-256 of model.py so any change to the model is visible.
Run by .github/workflows/live.yml, so the GitHub commit time is the timestamp.
"""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import model

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LOG = ROOT / "forecasts" / "live_log.csv"
NUMERIC = ["full", "weather", "avg5", "consensus_bcf", "actual_bcf"]
COLUMNS = ["week_ending", "logged_at_utc", "model_sha256", "full", "weather", "pb_gap",
           "avg5", "consensus_bcf", "actual_bcf"]
CONSENSUS_URL = ("https://endpoints.investing.com/pd-instruments/v1/calendars/economic/"
                 "events/386/occurrences?domain_id=1&limit=20")


def model_hash() -> str:
    return hashlib.sha256((ROOT / "model.py").read_bytes()).hexdigest()


def read_log() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG, parse_dates=["week_ending"])
    return pd.DataFrame(columns=COLUMNS)


def next_week_forecast(storage: pd.DataFrame, dd: pd.DataFrame, pb: pd.DataFrame):
    """Forecast for the week after the last EIA release. Returns None until all seven
    days of degree days for that week are in. Uses only data dated on or before the
    Friday that ends the target week."""
    s = storage.copy()
    s["week_ending"] = pd.to_datetime(s["week_ending"])
    target = s["week_ending"].max() + pd.Timedelta(7, "D")
    d = dd.copy()
    d["date"] = pd.to_datetime(d["date"])
    p = pb.copy()
    p["date"] = pd.to_datetime(p["date"])
    d, p = d[d["date"] <= target], p[p["date"] <= target]
    days = d[(d["date"] > target - pd.Timedelta(7, "D")) & d[["hdd", "cdd"]].notna().all(axis=1)]
    if len(days) < 7:
        return None
    # placeholder row for the target week; its change is never used as an input
    s = pd.concat([s, pd.DataFrame({"week_ending": [target], "storage_bcf": [np.nan],
                                    "change_bcf": [0.0]})], ignore_index=True)
    df = model.build_features(s, d, p)
    x = df.iloc[-1]
    hist = df.iloc[:-1]
    out = {"week_ending": target}
    for name in ["full", "weather"]:
        feats = model.FEATURES[name]
        train = hist.dropna(subset=feats + ["change_bcf"]).tail(model.TRAIN_WEEKS)
        ok = x[feats].notna().all() and len(train) == model.TRAIN_WEEKS
        out[name] = model._fit_predict(train, x, feats) if ok else np.nan
    out["pb_gap"] = bool(np.isnan(out["full"]))
    if out["pb_gap"]:
        out["full"] = out["weather"]
    out["avg5"] = x["avg5"]
    return out


def forecast():
    storage = pd.read_csv(DATA / "storage.csv")
    dd = pd.read_csv(DATA / "hdd_cdd.csv")
    pb = pd.read_csv(DATA / "power_burn.csv")
    f = next_week_forecast(storage, dd, pb)
    if f is None:
        print("degree days for the next report week are not complete yet; nothing logged")
        return
    log = read_log()
    if (log["week_ending"] == f["week_ending"]).any():
        print(f"week ending {f['week_ending'].date()} already logged; left unchanged")
        return
    row = {**f, "logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "model_sha256": model_hash(), "consensus_bcf": np.nan, "actual_bcf": np.nan}
    new = pd.DataFrame([row])[COLUMNS]
    log = new if log.empty else pd.concat([log, new], ignore_index=True)
    LOG.parent.mkdir(exist_ok=True)
    log.round({c: 2 for c in NUMERIC}).to_csv(LOG, index=False, date_format="%Y-%m-%d")
    print(f"logged week ending {f['week_ending'].date()}: full {f['full']:.1f}, "
          f"weather {f['weather']:.1f}, 5-yr avg {f['avg5']:.1f} Bcf")


def fetch_recent_consensus() -> pd.DataFrame:
    import requests
    r = requests.get(CONSENSUS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = [o for o in r.json()["occurrences"] if "actual" in o and "forecast" in o]
    c = pd.DataFrame({"release_date": [o["occurrence_time"][:10] for o in rows],
                      "consensus_bcf": [o["forecast"] for o in rows]})
    c["week_ending"] = model.release_to_week(c["release_date"])
    return c.drop_duplicates("week_ending", keep=False)


def score():
    log = read_log()
    if log.empty:
        print("no live forecasts yet")
        return
    storage = pd.read_csv(DATA / "storage.csv", parse_dates=["week_ending"])
    actual = storage.set_index("week_ending")["change_bcf"]
    todo = log["actual_bcf"].isna()
    log.loc[todo, "actual_bcf"] = log.loc[todo, "week_ending"].map(actual)
    try:
        cons = fetch_recent_consensus().set_index("week_ending")["consensus_bcf"]
        todo = log["consensus_bcf"].isna()
        log.loc[todo, "consensus_bcf"] = log.loc[todo, "week_ending"].map(cons)
    except Exception as e:  # source unavailable: leave blank, retried next run
        print(f"consensus not updated: {e}")
    log.round({c: 2 for c in NUMERIC}).to_csv(LOG, index=False, date_format="%Y-%m-%d")

    done = log.dropna(subset=["actual_bcf"])
    print(f"live weeks scored: {len(done)}")
    if len(done):
        for col in ["full", "weather", "avg5", "consensus_bcf"]:
            sub = done.dropna(subset=[col])
            if len(sub):
                print(f"  {col:14s} MAE {(sub['actual_bcf'] - sub[col]).abs().mean():6.2f} Bcf "
                      f"over {len(sub)} weeks")
        both = done.dropna(subset=["consensus_bcf"])
        gap = both["full"] - both["consensus_bcf"]
        both = both[gap != 0]
        if len(both):
            hit = (np.sign(both["full"] - both["consensus_bcf"])
                   == np.sign(both["actual_bcf"] - both["consensus_bcf"])).mean()
            print(f"  print on the model's side of consensus: {hit:.3f} over {len(both)} weeks")


if __name__ == "__main__":
    {"forecast": forecast, "score": score}[sys.argv[1]]()
