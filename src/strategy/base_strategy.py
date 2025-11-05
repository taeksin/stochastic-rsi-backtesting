from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals on top of OHLCV data."""

    @abstractmethod
    def calculate_position_size(self, capital: float, leverage: int) -> float:
        """Return notional position size for a new trade."""
