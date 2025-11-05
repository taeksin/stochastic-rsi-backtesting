from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return float("nan")
    mean = returns.mean()
    std = returns.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return (mean * periods_per_year) / (std * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    return drawdown.min()


def calculate_win_rate(returns: pd.Series) -> float:
    trades = returns[returns != 0]
    if trades.empty:
        return float("nan")
    wins = (trades > 0).sum()
    return wins / len(trades)
