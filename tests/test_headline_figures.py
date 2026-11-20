"The figures quoted in the README and the Overview page."

import pytest

from content.intro import compute_headline_figures


@pytest.fixture(scope="module")
def figures(merged):
    return compute_headline_figures(merged)


def test_underperforming_stores(figures):
    assert figures["under_stores"] == [9, 15, 21, 25, 29, 44, 45]
    assert figures["upside_per_year"] == pytest.approx(139e6, rel=0.01)


def test_growth(figures):
    assert figures["declining_stores"] == 18
    assert figures["total_growth"] == pytest.approx(0.019, abs=0.001)
    assert figures["store_14_rank"] == 3
    assert figures["store_14_growth"] == pytest.approx(-0.154, abs=0.001)


def test_holiday_peak(figures):
    assert figures["pre_christmas_lift"] == pytest.approx(0.75, abs=0.005)
    assert figures["thanksgiving_lift"] == pytest.approx(0.44, abs=0.005)


def test_departments(figures):
    assert figures["top10_dept_share"] == pytest.approx(0.46, abs=0.005)
    assert figures["growth_depts"] == [92, 87, 90]
    assert figures["shrinking_depts"] == [55, 6]


def test_annual_revenue(figures):
    assert figures["annual_revenue"] == pytest.approx(2.45e9, rel=0.005)
