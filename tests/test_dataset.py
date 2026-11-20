"Data quality checks on the merged dataset and its rebuild from the raw files."

from scripts.build_dataset import build


def test_shape_and_coverage(merged):
    assert len(merged) == 421_570
    assert merged["Store"].nunique() == 45
    assert merged["Dept"].nunique() == 81
    assert merged["Date"].nunique() == 143
    assert str(merged["Date"].min().date()) == "2010-02-05"
    assert str(merged["Date"].max().date()) == "2012-10-26"


def test_every_date_is_a_friday(merged):
    assert (merged["Date"].dt.dayofweek == 4).all()


def test_no_duplicate_store_dept_week(merged):
    assert not merged.duplicated(["Store", "Dept", "Date"]).any()


def test_join_left_no_gaps(merged):
    cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Type", "Size"]
    assert merged[cols].notna().all().all()


def test_markdowns_start_in_november_2011(merged):
    markdown = merged[[f"MarkDown{i}" for i in range(1, 6)]].sum(axis=1)
    assert merged.loc[markdown > 0, "Date"].min().strftime("%Y-%m-%d") == "2011-11-11"


def test_rebuild_matches_committed_file(merged):
    built = build()
    assert built.shape == merged.shape
    assert list(built.columns) == list(merged.columns)
    assert abs(built["Weekly_Sales"].sum() - merged["Weekly_Sales"].sum()) < 1
