"""
Run the SQL analyses in sql/ with DuckDB, straight on the raw CSV files.

    python scripts/run_sql.py            # run every query and print the results
    python scripts/run_sql.py 03         # run one query (by number)
    python scripts/run_sql.py --check    # check that SQL and the app give the same figures
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"


def connect():
    "Open an in-memory DuckDB database with the staging views loaded."
    con = duckdb.connect()
    con.execute(f"SET file_search_path = '{ROOT}'")
    con.execute((SQL_DIR / "00_staging.sql").read_text())
    return con


def run(con, prefix=""):
    "Print the result of every query whose file name starts with the prefix."
    for path in sorted(SQL_DIR.glob(f"{prefix}*.sql")):
        if path.name.startswith("00_"):
            continue
        print(f"\n=== {path.name} ===")
        print(con.sql(path.read_text()))


def check(con):
    "Compare SQL results with the figures shown in the Streamlit app."
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from content.intro import compute_headline_figures

    df = pd.read_csv(ROOT / "data" / "merged_retail_data.csv", parse_dates=["Date"])
    app = compute_headline_figures(df)

    productivity = con.sql((SQL_DIR / "01_store_productivity.sql").read_text()).df()
    under = sorted(productivity.loc[productivity["gap_vs_type"] < -0.25, "store_id"].tolist())
    assert under == app["under_stores"], (under, app["under_stores"])

    growth = con.sql((SQL_DIR / "02_store_growth.sql").read_text()).df()
    declining = int((growth["growth"] < 0).sum())
    assert declining == app["declining_stores"], (declining, app["declining_stores"])

    peak = con.sql((SQL_DIR / "03_holiday_peak.sql").read_text()).df()
    top_lift = peak["lift_vs_median_week"].max()
    assert abs(top_lift - app["pre_christmas_lift"]) < 0.005, (top_lift, app["pre_christmas_lift"])

    print(f"OK: underperformers {under}, {declining} declining stores, peak week +{top_lift:.0%}")


if __name__ == "__main__":
    connection = connect()
    if "--check" in sys.argv:
        check(connection)
    else:
        run(connection, sys.argv[1] if len(sys.argv) > 1 else "")
