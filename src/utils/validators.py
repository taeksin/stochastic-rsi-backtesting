from __future__ import annotations

from typing import Iterable, Set

import pandas as pd


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing: Set[str] = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
