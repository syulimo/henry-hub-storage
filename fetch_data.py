"""Fetch inputs for the Henry Hub storage model.

Outputs (data/):
  storage.csv     : weekly Lower 48 working gas, Bcf, week ending Friday (EIA)
  hdd_cdd.csv     : daily CONUS degree days (NOAA CPC): population-weighted HDD
                    and CDD, plus HDD weighted by homes heated with utility gas
  power_burn.csv  : daily Lower 48 natural gas-fired generation, MWh (EIA-930),
                    the proxy for gas burned in power plants
  hdd_regions.csv : daily population-weighted HDD for the nine Census divisions
                    (CPC); fetched only with the "regions" argument
  futures.csv     : daily NYMEX Henry Hub futures settlement, contracts 1 and 2,
                    $/MMBtu (EIA series RNGC1, RNGC2); fetched only with the
                    "futures" argument so the model inputs above stay as committed

Run: EIA_API_KEY=xxxx python fetch_data.py            (model inputs)
     EIA_API_KEY=xxxx python fetch_data.py futures    (futures prices only)
     python fetch_data.py regions                     (Census division HDD only, no key)
Get a free key at https://www.eia.gov/opendata/register.php
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

EIA_SERIES = "NW2_EPG0_SWO_R48_BCF"   # Lower 48 working gas in storage, weekly
EIA_URL = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

# NOAA CPC population-weighted degree days, daily, CONUS. Files are yearly.
# Population.*.txt holds the nine Census divisions plus a CONUS row;
# StatesCONUS.*.txt holds the 48 states only and has no CONUS row.
CPC_BASE = ("https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/"
            "daily_data/{year}/")
CPC_FILES = {"hdd": "Population.Heating.txt", "cdd": "Population.Cooling.txt",
             "hdd_gas": "UtilityGas.Heating.txt"}

# EIA-930 daily net generation by fuel type; US48 total, natural gas, Central time
EIA930_URL = "https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/"


def fetch_storage(api_key: str, start: str = "2010-01-01") -> pd.DataFrame:
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": EIA_SERIES,
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(EIA_URL, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    df = pd.DataFrame(rows)[["period", "value"]]
    df.columns = ["week_ending", "storage_bcf"]
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df["storage_bcf"] = pd.to_numeric(df["storage_bcf"])
    df = df.sort_values("week_ending").reset_index(drop=True)
    df["change_bcf"] = df["storage_bcf"].diff()
    return df


def _parse_cpc(text: str, name: str) -> pd.DataFrame:
    """CPC files: header rows, then 'Region|' lines; the CONUS line is what we want.
    Columns are dates as YYYYMMDD separated by '|'."""
    lines = [l for l in text.splitlines() if "|" in l]
    header = None
    conus = None
    for l in lines:
        cells = [c.strip() for c in l.split("|")]
        if cells[0].lower().startswith("region"):
            header = cells[1:]
        elif cells[0].upper() in ("CONUS", "CONTIGUOUS US", "US"):
            conus = cells[1:]
    if header is None or conus is None:
        raise ValueError(f"CONUS row not found in CPC {name} file; inspect format")
    dates = pd.to_datetime(header, format="%Y%m%d", errors="coerce")
    vals = pd.to_numeric(conus, errors="coerce")
    out = pd.DataFrame({"date": dates, name: vals}).dropna(subset=["date"])
    return out


def fetch_degree_days(years) -> pd.DataFrame:
    frames = []
    for y in years:
        parts = []
        for name, fname in CPC_FILES.items():
            url = CPC_BASE.format(year=y) + fname
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"skip {url}: {r.status_code}", file=sys.stderr)
                continue
            parts.append(_parse_cpc(r.text, name))
        if len(parts) == len(CPC_FILES):
            year = parts[0]
            for p in parts[1:]:
                year = year.merge(p, on="date", how="outer")
            frames.append(year)
    df = pd.concat(frames).sort_values("date").drop_duplicates("date")
    return df.reset_index(drop=True)


def fetch_power_burn(api_key: str, start: str = "2018-07-01") -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[respondent][]": "US48",
            "facets[fueltype][]": "NG",
            "facets[timezone][]": "Central",
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": 5000,
        }
        r = requests.get(EIA930_URL, params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()["response"]["data"]
        rows += batch
        if len(batch) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)[["period", "value"]]
    df.columns = ["date", "gas_gen_mwh"]
    df["date"] = pd.to_datetime(df["date"])
    df["gas_gen_mwh"] = pd.to_numeric(df["gas_gen_mwh"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


EIA_FUT_URL = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"


def fetch_futures(api_key: str, start: str = "2017-01-01") -> pd.DataFrame:
    out = None
    for series, col in [("RNGC1", "c1"), ("RNGC2", "c2")]:
        rows, offset = [], 0
        while True:
            params = {
                "api_key": api_key,
                "frequency": "daily",
                "data[0]": "value",
                "facets[series][]": series,
                "start": start,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": 5000,
            }
            r = requests.get(EIA_FUT_URL, params=params, timeout=60)
            r.raise_for_status()
            batch = r.json()["response"]["data"]
            rows += batch
            if len(batch) < 5000:
                break
            offset += 5000
        df = pd.DataFrame(rows)[["period", "value"]]
        df.columns = ["date", col]
        df["date"] = pd.to_datetime(df["date"])
        df[col] = pd.to_numeric(df[col])
        df = df.drop_duplicates("date")
        out = df if out is None else out.merge(df, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def fetch_hdd_regions(years) -> pd.DataFrame:
    frames = []
    for y in years:
        url = CPC_BASE.format(year=y) + CPC_FILES["hdd"]
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"skip {url}: {r.status_code}", file=sys.stderr)
            continue
        rows = [[c.strip() for c in l.split("|")] for l in r.text.splitlines() if "|" in l]
        header = next(c[1:] for c in rows if c[0].lower().startswith("region"))
        year = pd.DataFrame({"date": pd.to_datetime(header, format="%Y%m%d", errors="coerce")})
        for c in rows:
            if c[0].isdigit():
                year[f"hdd_div{c[0]}"] = pd.to_numeric(c[1:], errors="coerce")
        frames.append(year.dropna(subset=["date"]))
    return pd.concat(frames).sort_values("date").drop_duplicates("date").reset_index(drop=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["regions"]:
        years = range(2010, pd.Timestamp.today().year + 1)
        reg = fetch_hdd_regions(years)
        reg.to_csv(DATA / "hdd_regions.csv", index=False)
        print(f"regional HDD: {len(reg)} days, {reg.date.min().date()} to {reg.date.max().date()}")
        sys.exit(0)
    key = os.environ.get("EIA_API_KEY")
    if not key:
        sys.exit("set EIA_API_KEY")
    if sys.argv[1:] == ["futures"]:
        fut = fetch_futures(key)
        fut.to_csv(DATA / "futures.csv", index=False)
        print(f"futures: {len(fut)} days, {fut.date.min().date()} to {fut.date.max().date()}")
        sys.exit(0)
    storage = fetch_storage(key)
    storage.to_csv(DATA / "storage.csv", index=False)
    print(f"storage: {len(storage)} weeks, {storage.week_ending.min().date()} to "
          f"{storage.week_ending.max().date()}")
    years = range(storage.week_ending.min().year, storage.week_ending.max().year + 1)
    dd = fetch_degree_days(years)
    dd.to_csv(DATA / "hdd_cdd.csv", index=False)
    print(f"degree days: {len(dd)} days, {dd.date.min().date()} to {dd.date.max().date()}")
    pb = fetch_power_burn(key)
    pb.to_csv(DATA / "power_burn.csv", index=False)
    print(f"power burn: {len(pb)} days, {pb.date.min().date()} to {pb.date.max().date()}")
