"""
Build data/merged_retail_data.csv from the three raw CSV files.

    python scripts/build_dataset.py           # rebuild the merged file
    python scripts/build_dataset.py --check   # check that the committed file matches a rebuild

Steps (same as the Data Processing page):
1. MarkDown1-5: missing values mean "no promotion", filled with 0
2. Dates parsed from DD/MM/YYYY (each date is the Friday ending the week)
3. CPI and Unemployment: forward then back fill within each store
4. sales LEFT JOIN features ON (Store, Date), then LEFT JOIN stores ON Store
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MERGED = DATA / "merged_retail_data.csv"
MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]


def build():
    "Return the merged dataset built from the raw files."
    features = pd.read_csv(DATA / "features.csv")
    sales = pd.read_csv(DATA / "sales.csv")
    stores = pd.read_csv(DATA / "stores.csv")

    features[MARKDOWN_COLS] = features[MARKDOWN_COLS].fillna(0)
    features["Date"] = pd.to_datetime(features["Date"], format="%d/%m/%Y")
    sales["Date"] = pd.to_datetime(sales["Date"], format="%d/%m/%Y")

    features = features.sort_values(["Store", "Date"])
    for method in ("ffill", "bfill"):
        features[["CPI", "Unemployment"]] = getattr(
            features.groupby("Store", group_keys=False)[["CPI", "Unemployment"]], method
        )()

    merged = sales.merge(features, on=["Store", "Date"], how="left")
    return merged.merge(stores, on="Store", how="left")


def check():
    "Compare a fresh build with the committed file."
    built = build()
    committed = pd.read_csv(MERGED, parse_dates=["Date"])
    pd.testing.assert_frame_equal(built, committed, check_dtype=False, rtol=1e-9)
    print(f"OK: {MERGED.name} matches a rebuild ({len(built):,} rows, {built.shape[1]} columns)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        build().to_csv(MERGED, index=False)
        print(f"Written {MERGED}")
