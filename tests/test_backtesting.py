from __future__ import annotations

import pandas as pd

import pytest

from src.backtesting.engine import BacktestEngine
from src.strategy.ma200_stochrsi import MA200StochRSIStrategy
from src.strategy.base_strategy import BaseStrategy


def sample_price_data(size: int = 300) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=size, freq="5T")
    price = pd.Series(range(size), dtype=float) + 100
    return pd.DataFrame({"timestamp": dates, "open": price, "high": price, "low": price, "close": price, "volume": 1})


def test_engine_returns_report_structure():
    engine = BacktestEngine()
    strategy = MA200StochRSIStrategy()
    df = sample_price_data()

    report = engine.run(
        df,
        strategy,
        initial_capital=1000,
        leverage=2,
        take_profit_pct=0.02,
        stop_loss_pct=0.015,
    )

    assert not report.equity_curve.empty
    assert {
        "total_return",
        "final_equity",
        "trade_capital",
        "take_profit_pct",
        "stop_loss_pct",
        "initial_capital",
    } <= report.metrics.keys()
    assert isinstance(report.trades, pd.DataFrame)
    if not report.trades.empty:
        assert {"entry_time", "exit_time", "pnl_pct", "pnl_value", "balance"}.issubset(
            report.trades.columns
        )


def test_engine_uses_default_trade_capital():
    engine = BacktestEngine(default_trade_capital=150)
    strategy = MA200StochRSIStrategy()
    df = sample_price_data()

    report = engine.run(
        df,
        strategy,
        initial_capital=1000,
        leverage=1,
        take_profit_pct=0.02,
        stop_loss_pct=0.015,
    )

    assert report.metrics.get("trade_capital") == 150


class DummyStrategy(BaseStrategy):
    def __init__(self, signals, positions):
        self._signals = signals
        self._positions = positions

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = pd.Series(self._signals, index=df.index)
        result["position"] = pd.Series(self._positions, index=df.index)
        return result

    def calculate_position_size(self, capital: float, leverage: int) -> float:
        return capital * leverage


def test_take_profit_applies_leverage_and_fees():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 105.0, 100.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 104.0, 100.0],
            "volume": [1, 1, 1],
        }
    )

    strategy = DummyStrategy(signals=[1, 0, 0], positions=[1, 1, 0])
    engine = BacktestEngine()
    report = engine.run(
        df,
        strategy=strategy,
        initial_capital=1000.0,
        leverage=2,
        trade_capital=100.0,
        take_profit_pct=0.05,
        stop_loss_pct=0.02,
    )

    trade = report.trades.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert pytest.approx(trade["pnl_value"], rel=1e-4) == 9.84
    assert pytest.approx(report.metrics["final_equity"], rel=1e-4) == 1009.84


def test_stop_loss_applies_leverage_and_fees():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 103.5, 100.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 102.0, 100.0],
            "volume": [1, 1, 1],
        }
    )

    strategy = DummyStrategy(signals=[-1, 0, 0], positions=[-1, -1, 0])
    engine = BacktestEngine()
    report = engine.run(
        df,
        strategy=strategy,
        initial_capital=1000.0,
        leverage=3,
        trade_capital=200.0,
        take_profit_pct=0.05,
        stop_loss_pct=0.03,
    )

    trade = report.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert pytest.approx(trade["pnl_value"], rel=1e-4) == -18.48
    assert pytest.approx(report.metrics["final_equity"], rel=1e-4) == 981.52
