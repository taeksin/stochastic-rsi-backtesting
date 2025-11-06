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
    cooldown_minutes: int = 60


class MA200StochRSIStrategy(BaseStrategy):
    def __init__(self, params: Optional[StrategyParameters] = None):
        self.params = params or StrategyParameters()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ensure_columns(df, ["close"])
        result = df.copy()
        has_timestamp = "timestamp" in result.columns
        if has_timestamp:
            result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

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

        ma_trend_valid = result["ma_trend"].notna()
        bias = np.where(
            ma_trend_valid & (result["close"] >= result["ma_trend"]),
            1,
            np.where(ma_trend_valid, -1, 0),
        )
        bias_series = pd.Series(bias, index=result.index)
        result["ma_bias"] = bias_series
        bias_change = bias_series.ne(bias_series.shift(1)).fillna(False) & (bias_series != 0)
        result["ma_bias_change"] = bias_change

        cooldown_mask_long = pd.Series(False, index=result.index)
        cooldown_mask_short = pd.Series(False, index=result.index)
        cooldown_minutes = max(int(getattr(self.params, "cooldown_minutes", 0) or 0), 0)
        if cooldown_minutes > 0 and has_timestamp:
            timestamps = result["timestamp"]
            cooldown_delta = pd.to_timedelta(cooldown_minutes, unit="m")
            change_times = timestamps.where(bias_change, pd.NaT)
            last_change = change_times.ffill()
            cooldown_until = last_change + cooldown_delta
            cooldown_active = (
                cooldown_until.notna()
                & timestamps.notna()
                & (cooldown_until > timestamps)
                & (bias_series != 0)
            )
            result["cooldown_until"] = cooldown_until
            result["cooldown_active"] = cooldown_active.fillna(False)
            cooldown_mask_long = cooldown_active & (bias_series > 0)
            cooldown_mask_short = cooldown_active & (bias_series < 0)

        else:
            result["cooldown_active"] = False

        long_entry = (
            (result["close"] > result["ma_trend"])
            & result["golden_cross"]
            & (result["stoch_d"] < self.params.oversold)
            & ~cooldown_mask_long
        )
        short_entry = (
            (result["close"] < result["ma_trend"])
            & result["dead_cross"]
            & (result["stoch_d"] > self.params.overbought)
            & ~cooldown_mask_short
        )

        signals = np.zeros(len(result), dtype=int)
        signals[long_entry.fillna(False)] = 1
        signals[short_entry.fillna(False)] = -1
        result["signal"] = signals

        position = []
        exit_signals = []
        current = 0
        cooldown_active_col = "cooldown_active" in result.columns
        for _, row in result.iterrows():
            exit_signal = ""
            signal_value = int(row["signal"])

            if signal_value == 1:
                if current == -1:
                    exit_signal = "reverse_to_long"
                current = 1
            elif signal_value == -1:
                if current == 1:
                    exit_signal = "reverse_to_short"
                current = -1
            else:
                if current == 1 and bool(row.get("dead_cross", False)):
                    current = 0
                    exit_signal = "dead_cross_exit"
                elif current == -1 and bool(row.get("golden_cross", False)):
                    current = 0
                    exit_signal = "golden_cross_exit"
                elif current != 0 and cooldown_active_col and bool(row.get("cooldown_active", False)):
                    current = 0
                    exit_signal = "cooldown_exit"
                elif current != 0 and row.get("ma_bias", 0) == 0:
                    current = 0
                    exit_signal = "bias_neutral_exit"

            position.append(current)
            exit_signals.append(exit_signal)

        result["position"] = position
        result["exit_signal"] = exit_signals
        return result

    def calculate_position_size(self, capital: float, leverage: int) -> float:
        return capital * leverage
