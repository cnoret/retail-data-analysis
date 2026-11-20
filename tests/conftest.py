import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def merged():
    return pd.read_csv(ROOT / "data" / "merged_retail_data.csv", parse_dates=["Date"])
