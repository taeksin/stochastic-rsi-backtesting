from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtesting.engine import BacktestReport
from src.utils import storage


def sample_report() -> BacktestReport:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="H")
    equity_curve = pd.Series([1000, 1010], index=timestamps)
    returns = pd.Series([0.0, 0.01], index=timestamps)
    metrics = {"total_return": 0.01}
    trades = pd.DataFrame(
        {
            "entry_index": [0],
            "exit_index": [1],
            "direction": ["long"],
            "pnl_value": [10.0],
            "exit_reason": ["take_profit_long"],
            "exit_price": [102.0],
        }
    )
    price_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 12],
            "signal": [1, 0],
            "position": [1, 0],
            "exit_signal": ["", "dead_cross_exit"],
        }
    )
    return BacktestReport(
        equity_curve=equity_curve,
        returns=returns,
        metrics=metrics,
        trades=trades,
        price_frame=price_frame,
    )


def test_save_backtest_result_handles_missing_trade_columns(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(storage, "_root_dir", lambda: tmp_path)

    report = sample_report()
    price_df = report.price_frame

    context = {"parameters": {"start_date": "2024-01-01", "end_date": "2024-01-02"}}

    record_id = storage.save_backtest_result(report, price_df, context)

    file_path = tmp_path / "storage" / "backtests" / f"{record_id}.json"
    assert file_path.exists()

    data = json.loads(file_path.read_text(encoding="utf-8"))
    files = data["files"]
    price_csv = tmp_path / "storage" / "backtests" / files["price_csv"]
    equity_csv = tmp_path / "storage" / "backtests" / files["equity_curve_csv"]
    trades_csv = tmp_path / "storage" / "backtests" / files["trades_csv"]
    assert price_csv.exists()
    assert equity_csv.exists()
    assert trades_csv.exists()


def test_list_and_load_backtests(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(storage, "_root_dir", lambda: tmp_path)

    report = sample_report()
    price_df = report.price_frame

    context = {
        "parameters": {"start_date": "2024-01-01", "end_date": "2024-01-02", "timeframe": "5T"},
        "strategy": {"ma_period": 200},
    }

    record_id = storage.save_backtest_result(report, price_df, context)

    records = storage.list_backtests()
    assert any(r["id"] == record_id for r in records)

    loaded = storage.load_backtest(record_id)
    assert loaded is not None
    price_loaded = storage.load_price_csv(loaded)
    assert not price_loaded.empty
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
