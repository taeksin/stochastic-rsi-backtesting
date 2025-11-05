from __future__ import annotations

import pandas as pd

from src.utils.validators import ensure_columns


class KlineResampler:
    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        ensure_columns(df, KlineResampler.REQUIRED_COLUMNS)
        data = df.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.sort_values("timestamp")
        return data

    @staticmethod
    def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        validated = KlineResampler.validate_data(df)
        resampled = (
            validated.set_index("timestamp")
            .resample(timeframe, label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
        return resampled
