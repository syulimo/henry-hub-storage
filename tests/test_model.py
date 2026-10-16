"""Tests run on synthetic data, no network needed: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402


def synthetic(n_weeks=680, seed=0):
    rng = np.random.default_rng(seed)
    fridays = pd.date_range("2015-01-02", periods=n_weeks, freq="W-FRI")
    days = pd.date_range(fridays[0] - pd.Timedelta(6, "D"), fridays[-1], freq="D")
    doy = days.dayofyear.values
    hdd = np.clip(20 * np.cos(2 * np.pi * doy / 365) + 15 + rng.normal(0, 4, len(days)), 0, None)
    cdd = np.clip(-10 * np.cos(2 * np.pi * doy / 365) + 5 + rng.normal(0, 2, len(days)), 0, None)
    gen = 4e6 + 1.2e5 * cdd + rng.normal(0, 6e5, len(days))
    dd = pd.DataFrame({"date": days, "hdd": hdd, "cdd": cdd})
    pb = pd.DataFrame({"date": days, "gas_gen_mwh": gen})
    w = dd.assign(gen=gen, week_ending=model.week_ending(dd["date"]))
    w = w.groupby("week_ending")[["hdd", "cdd", "gen"]].sum().reindex(fridays)
    change = (40 - 0.9 * w["hdd"].values + 0.004 * w["hdd"].values ** 2
              - 5.0 * (w["gen"].values / 1e6 - 28) + rng.normal(0, 6, n_weeks))
    storage = pd.DataFrame({"week_ending": fridays, "storage_bcf": 2500 + np.cumsum(change)})
    storage["change_bcf"] = storage["storage_bcf"].diff()
    return storage, dd, pb


def _preds(storage, dd, pb):
    return model.run_all(model.build_features(storage, dd, pb)).set_index("week_ending")


def test_weeks_align_to_friday():
    storage, dd, pb = synthetic()
    df = model.build_features(storage, dd, pb)
    assert (df["week_ending"].dt.weekday == 4).all()


def test_no_lookahead_in_degree_days():
    """Changing degree days after week t must not change any forecast for week t."""
    storage, dd, pb = synthetic()
    a = _preds(storage, dd, pb)
    cut = a.index[len(a) // 2]
    dd2 = dd.copy()
    dd2.loc[dd2["date"] > cut, ["hdd", "cdd"]] *= 3
    b = _preds(storage, dd2, pb)
    for col in ["full", "weather", "v1"]:
        assert np.allclose(a.loc[:cut, col], b.loc[:cut, col]), col


def test_no_lookahead_in_power_burn():
    """Changing generation after week t, including a gap the cleaner would drop,
    must not change any forecast for week t."""
    storage, dd, pb = synthetic()
    a = _preds(storage, dd, pb)
    cut = a.index[len(a) // 2]
    pb2 = pb.copy()
    pb2.loc[pb2["date"] > cut, "gas_gen_mwh"] *= 0.2
    b = _preds(storage, dd, pb2)
    assert np.allclose(a.loc[:cut, "full"], b.loc[:cut, "full"])


def test_no_lookahead_in_storage():
    """Changing storage after week t must not change forecasts or avg5 for week t."""
    storage, dd, pb = synthetic()
    a = _preds(storage, dd, pb)
    cut = a.index[len(a) // 2]
    s2 = storage.copy()
    s2.loc[s2["week_ending"] > cut, "change_bcf"] += 500
    b = _preds(s2, dd, pb)
    for col in ["full", "weather", "v1", "avg5"]:
        assert np.allclose(a.loc[:cut, col], b.loc[:cut, col]), col


def test_power_burn_gap_is_dropped():
    days = pd.date_range("2020-01-01", periods=800, freq="D")
    rng = np.random.default_rng(1)
    pb = pd.DataFrame({"date": days, "gas_gen_mwh": 4e6 + rng.normal(0, 2e5, len(days))})
    pb.loc[pb.index[400:405], "gas_gen_mwh"] *= 0.3
    pb.loc[pb.index[500:560], "gas_gen_mwh"] *= 0.4   # a gap longer than the 28-day window
    cleaned = model.clean_power_burn(pb)
    assert cleaned.loc[pb.index[400:405], "gas_gen_mwh"].isna().all()
    assert cleaned.loc[pb.index[500:560], "gas_gen_mwh"].isna().all()
    assert cleaned["gas_gen_mwh"].isna().sum() == 65


def test_models_beat_benchmarks_when_signal_exists():
    storage, dd, pb = synthetic()
    pred = model.run_all(model.build_features(storage, dd, pb))
    s = model.score(pred, ["full", "weather", "v1", "naive", "avg5"])
    assert s.loc["full", "MAE_bcf"] < s.loc["weather", "MAE_bcf"] < s.loc["v1", "MAE_bcf"]
    assert s.loc["full", "MAE_bcf"] < s.loc["avg5", "MAE_bcf"]
    assert s.loc["full", "MAE_bcf"] < s.loc["naive", "MAE_bcf"]


def test_avg5_uses_only_prior_years():
    storage, dd, pb = synthetic()
    df = model.build_features(storage, dd, pb)
    row = df.dropna(subset=["avg5"]).iloc[-1]
    prior = df[(df["woy"] == row["woy"]) & (df["week_ending"] < row["week_ending"])]
    assert np.isclose(row["avg5"], prior["change_bcf"].tail(5).mean())


def test_release_maps_to_report_week_in_holiday_weeks():
    rel = pd.Series(pd.to_datetime(["2026-10-15", "2025-11-26", "2025-11-14", "2025-12-31"]))
    wk = model.release_to_week(rel)
    # Thursday, Wednesday (Thanksgiving), Friday (delayed), Wednesday (New Year)
    assert list(wk.dt.strftime("%Y-%m-%d")) == ["2026-10-09", "2025-11-21", "2025-11-07", "2025-12-26"]


def test_blend_uses_only_past_weeks():
    """Changing actuals after week t must not change the blend for week t."""
    rng = np.random.default_rng(3)
    n = 120
    wk = pd.date_range("2022-01-07", periods=n, freq="W-FRI")
    cons = rng.normal(0, 50, n)
    m = pd.DataFrame({"week_ending": wk, "consensus_bcf": cons,
                      "full": cons + rng.normal(0, 8, n), "actual": cons + rng.normal(0, 8, n)})
    a = model.add_blend(m)
    m2 = m.copy()
    m2.loc[80:, "actual"] += 300
    b = model.add_blend(m2)
    assert a["blend"].iloc[:52].isna().all()
    assert np.allclose(a["blend"].iloc[52:81], b["blend"].iloc[52:81])


import live  # noqa: E402


def test_live_forecast_matches_walk_forward():
    """The frozen live forecast for a week equals the backtest forecast for that week,
    and data after the target Friday cannot change it."""
    storage, dd, pb = synthetic()
    pred = model.run_all(model.build_features(storage, dd, pb)).set_index("week_ending")
    target = pred.index[-40]
    f = live.next_week_forecast(storage[storage["week_ending"] < target], dd, pb)
    assert f["week_ending"] == target
    assert np.isclose(f["full"], pred.loc[target, "full"])
    assert np.isclose(f["weather"], pred.loc[target, "weather"])
    dd2, pb2 = dd.copy(), pb.copy()
    dd2.loc[dd2["date"] > target, ["hdd", "cdd"]] *= 5
    pb2.loc[pb2["date"] > target, "gas_gen_mwh"] *= 5
    g = live.next_week_forecast(storage[storage["week_ending"] < target], dd2, pb2)
    assert np.isclose(f["full"], g["full"])


def test_live_waits_for_complete_week():
    storage, dd, pb = synthetic()
    target = storage["week_ending"].iloc[-40]
    dd_short = dd[dd["date"] < target - pd.Timedelta(1, "D")]
    assert live.next_week_forecast(storage[storage["week_ending"] < target], dd_short, pb) is None


def test_live_log_is_append_only(tmp_path, monkeypatch):
    storage, dd, pb = synthetic()
    target = storage["week_ending"].iloc[-40]
    data = tmp_path / "data"
    data.mkdir()
    storage[storage["week_ending"] < target].to_csv(data / "storage.csv", index=False)
    dd.to_csv(data / "hdd_cdd.csv", index=False)
    pb.to_csv(data / "power_burn.csv", index=False)
    monkeypatch.setattr(live, "DATA", data)
    monkeypatch.setattr(live, "LOG", tmp_path / "forecasts" / "live_log.csv")
    live.forecast()
    first = (tmp_path / "forecasts" / "live_log.csv").read_text()
    dd.assign(hdd=dd["hdd"] * 2).to_csv(data / "hdd_cdd.csv", index=False)
    live.forecast()
    assert (tmp_path / "forecasts" / "live_log.csv").read_text() == first
    assert len(pd.read_csv(tmp_path / "forecasts" / "live_log.csv")) == 1


import price_reaction  # noqa: E402


def _futures(days):
    return pd.DataFrame({"date": days, "c1": np.linspace(3.0, 3.5, len(days)),
                         "c2": np.linspace(3.2, 3.7, len(days))})


def test_last_trading_day_is_third_to_last_of_month():
    days = pd.bdate_range("2026-01-01", "2026-03-31")
    ltd = price_reaction.last_trading_days(pd.Series(days))
    assert pd.Timestamp("2026-01-28") in ltd   # Jan 29, 30 follow
    assert pd.Timestamp("2026-02-25") in ltd   # Feb 26, 27 follow


def test_release_return_uses_contract_two_across_the_roll():
    days = pd.bdate_range("2026-01-01", "2026-02-27")
    fut = _futures(days)
    r = price_reaction.release_returns(pd.Series([pd.Timestamp("2026-01-29"),
                                                  pd.Timestamp("2026-02-05")]), fut)
    f = fut.set_index("date")
    rolled = r.set_index("release_date").loc["2026-01-29"]
    assert bool(rolled["rolled"]) and np.isclose(rolled["start_price"], f.loc["2026-01-28", "c2"])
    plain = r.set_index("release_date").loc["2026-02-05"]
    assert not plain["rolled"] and np.isclose(plain["start_price"], f.loc["2026-02-04", "c1"])


def test_signal_position_is_set_before_release():
    """Position depends only on the model forecast and the consensus, never on the
    actual print or the release-day price."""
    m = pd.DataFrame({"week_ending": pd.to_datetime(["2026-01-02", "2026-01-09"]),
                      "actual": [10.0, 10.0], "full": [20.0, 0.0], "consensus_bcf": [5.0, 5.0]})
    cons = pd.DataFrame({"release_date": ["2026-01-08", "2026-01-15"], "actual_bcf": [10, 10],
                         "consensus_bcf": [5, 5]})
    fut = _futures(pd.bdate_range("2026-01-02", "2026-01-20"))
    a = price_reaction.build(m, cons, fut)
    m2 = m.assign(actual=[-500.0, 500.0])
    fut2 = fut.assign(c1=fut["c1"] * 2)
    b = price_reaction.build(m2, cons, fut2)
    assert list(a["position"]) == list(b["position"]) == [-1.0, 1.0]


def test_release_on_last_trading_day_uses_contract_two():
    days = pd.bdate_range("2026-01-01", "2026-02-27")
    fut = _futures(days)
    r = price_reaction.release_returns(pd.Series([pd.Timestamp("2026-01-28")]), fut).iloc[0]
    f = fut.set_index("date")
    assert np.isclose(r["start_price"], f.loc["2026-01-27", "c2"])
    assert np.isclose(r["end_price"], f.loc["2026-01-28", "c2"])


def test_intraday_window_and_expiry_exclusion():
    days = pd.Series(pd.bdate_range("2026-01-01", "2026-02-27"))
    h = pd.DataFrame({"release_date": ["2026-01-08", "2026-01-28", "2026-01-27", "2026-01-14"],
                      "open_0900": [3.0] * 4, "open_1000": [3.0, 3.0, 3.0, 3.0],
                      "close_1100": [3.3, 3.0, 3.0, 2.0], "close_1300": [9.0, 9.0, 9.0, 3.3]})
    r = price_reaction.intraday_returns(h, days).set_index("release_date")
    # Jan 28 is the last trading day and Jan 27 the day before it: both dropped
    assert list(r.index.strftime("%Y-%m-%d")) == ["2026-01-08", "2026-01-14"]
    assert np.isclose(r.loc["2026-01-08", "ret_intraday"], np.log(3.3 / 3.0))   # Thursday: 11:00
    assert np.isclose(r.loc["2026-01-14", "ret_intraday"], np.log(3.3 / 3.0))   # Wednesday: 13:00
