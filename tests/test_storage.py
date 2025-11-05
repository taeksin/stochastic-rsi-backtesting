from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtesting.engine import BacktestReport
from src.utils import storage


def sample_report() -> BacktestReport:
    equity_curve = pd.Series([1000, 1010])
    returns = pd.Series([0.0, 0.01])
    metrics = {"total_return": 0.01}
    trades = pd.DataFrame(
        {
            "entry_index": [0],
            "exit_index": [1],
            "direction": ["long"],
            "pnl_value": [10.0],
        }
    )
    return BacktestReport(
        equity_curve=equity_curve,
        returns=returns,
        metrics=metrics,
        trades=trades,
    )


def test_save_backtest_result_handles_missing_trade_columns(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(storage, "_root_dir", lambda: tmp_path)

    report = sample_report()
    price_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="H"),
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 12],
        }
    )

    context = {"parameters": {"start_date": "2024-01-01", "end_date": "2024-01-02"}}

    record_id = storage.save_backtest_result(report, price_df, context)

    file_path = tmp_path / "storage" / "backtests" / f"{record_id}.json"
    assert file_path.exists()

    data = json.loads(file_path.read_text(encoding="utf-8"))
    files = data["files"]
    equity_csv = tmp_path / "storage" / "backtests" / files["equity_curve_csv"]
    trades_csv = tmp_path / "storage" / "backtests" / files["trades_csv"]
    assert equity_csv.exists()
    assert trades_csv.exists()


def test_list_and_load_backtests(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(storage, "_root_dir", lambda: tmp_path)

    report = sample_report()
    price_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="H"),
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 12],
        }
    )

    context = {
        "parameters": {"start_date": "2024-01-01", "end_date": "2024-01-02", "timeframe": "5T"},
        "strategy": {"ma_period": 200},
    }

    record_id = storage.save_backtest_result(report, price_df, context)

    records = storage.list_backtests()
    assert any(r["id"] == record_id for r in records)

    loaded = storage.load_backtest(record_id)
    assert loaded is not None
    equity_series = storage.load_equity_curve_csv(loaded)
    assert not equity_series.empty
    trades_df = storage.load_trades_csv(loaded)
    assert not trades_df.empty


def test_load_trades_csv_handles_empty_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(storage, "_root_dir", lambda: tmp_path)

    storage_dir = tmp_path / "storage" / "backtests"
    storage_dir.mkdir(parents=True, exist_ok=True)
    empty_path = storage_dir / "empty_trades.csv"
    empty_path.write_text("entry_index,entry_time,exit_index,exit_time,direction,pnl_pct,pnl_value,balance,exit_reason,trade_capital\n", encoding="utf-8")

    meta = {"files": {"trades_csv": empty_path.name}}
    df = storage.load_trades_csv(meta)
    assert df.empty
