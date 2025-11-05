from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.indicators.moving_average import calculate_sma
from src.indicators import stochastic_rsi
from src.strategy.base_strategy import BaseStrategy
from src.utils.validators import ensure_columns


@dataclass(slots=True)
class StrategyParameters:
    ma_period: int = 200
    rsi_period: int = 14
    stoch_period: int = 14
    stoch_k: int = 3
    stoch_d: int = 3
    oversold: float = 20.0
    overbought: float = 80.0


class MA200StochRSIStrategy(BaseStrategy):
    def __init__(self, params: Optional[StrategyParameters] = None):
        self.params = params or StrategyParameters()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ensure_columns(df, ["close"])
        result = df.copy()

        result["ma_trend"] = calculate_sma(result, period=self.params.ma_period)
        stoch_df = stochastic_rsi.calculate(
            result,
            column="close",
            rsi_period=self.params.rsi_period,
            stoch_period=self.params.stoch_period,
            k=self.params.stoch_k,
            d=self.params.stoch_d,
        )
        result[["rsi", "stoch_rsi", "stoch_k", "stoch_d"]] = stoch_df[
            ["rsi", "stoch_rsi", "stoch_k", "stoch_d"]
        ]

        cross_df = stochastic_rsi.detect_crossover(result)
        result[["golden_cross", "dead_cross"]] = cross_df[
            ["golden_cross", "dead_cross"]
        ].fillna(False).astype(bool)

        long_entry = (
            (result["close"] > result["ma_trend"])
            & result["golden_cross"]
            & (result["stoch_d"] < self.params.oversold)
        )
        short_entry = (
            (result["close"] < result["ma_trend"])
            & result["dead_cross"]
            & (result["stoch_d"] > self.params.overbought)
        )

        signals = np.zeros(len(result), dtype=int)
        signals[long_entry.fillna(False)] = 1
        signals[short_entry.fillna(False)] = -1
        result["signal"] = signals

        position = []
        current = 0
        for _, row in result.iterrows():
            if row["signal"] == 1:
                current = 1
            elif row["signal"] == -1:
                current = -1
            elif current == 1 and row["dead_cross"]:
                current = 0
            elif current == -1 and row["golden_cross"]:
                current = 0
            position.append(current)

        result["position"] = position
        return result

    def calculate_position_size(self, capital: float, leverage: int) -> float:
        return capital * leverage
