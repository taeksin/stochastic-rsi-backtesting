from __future__ import annotations

import pandas as pd

from src.utils.validators import ensure_columns


def calculate_sma(df: pd.DataFrame, column: str = "close", period: int = 200) -> pd.Series:
    ensure_columns(df, [column])
    return df[column].rolling(window=period, min_periods=period).mean()


def calculate_ema(df: pd.DataFrame, column: str = "close", period: int = 200) -> pd.Series:
    ensure_columns(df, [column])
    return df[column].ewm(span=period, adjust=False, min_periods=period).mean()
