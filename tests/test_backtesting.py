from __future__ import annotations

import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.strategy.ma200_stochrsi import MA200StochRSIStrategy


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
