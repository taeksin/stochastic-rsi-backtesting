from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestReport


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _storage_dir() -> Path:
    path = _root_dir() / "storage" / "backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_python_number(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_backtest_result(
    report: BacktestReport,
    price_df: pd.DataFrame,
    context: Dict[str, Any],
) -> str:
    record_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex
    storage = _storage_dir()

    price_records = price_df.copy()
    if not price_records.empty:
        if "timestamp" in price_records:
            price_records["timestamp"] = price_records["timestamp"].astype(str)
        price_records = price_records.apply(lambda col: col.map(_to_python_number))
        price_records = price_records.where(price_records.notna(), None)

    equity_values = report.equity_curve.reset_index(drop=True).tolist()
    timestamps = (
        price_df["timestamp"].astype(str).tolist()
        if "timestamp" in price_df.columns
        else list(range(len(equity_values)))
    )
    equity_curve = []
    for ts, val in zip(timestamps, equity_values):
        equity_curve.append(
            {
                "timestamp": ts,
                "equity": _to_python_number(val),
            }
        )

    trades_records: List[Dict[str, Any]] = []
    if report.trades is not None and not report.trades.empty:
        trades_df = report.trades.copy()
        expected_columns = [
            "entry_index",
            "entry_time",
            "exit_index",
            "exit_time",
            "direction",
            "pnl_pct",
            "pnl_value",
            "balance",
            "exit_reason",
            "trade_capital",
        ]
        for column in expected_columns:
            if column not in trades_df.columns:
                trades_df[column] = None

        for column in ["entry_time", "exit_time"]:
            if column in trades_df.columns and trades_df[column].notna().any():
                trades_df[column] = pd.to_datetime(trades_df[column], errors="coerce")
                trades_df[column] = trades_df[column].dt.tz_localize(None)
                trades_df[column] = trades_df[column].astype(str)
            elif column in trades_df.columns:
                trades_df[column] = None

        trades_df = trades_df[expected_columns]
        trades_df = trades_df.apply(lambda col: col.map(_to_python_number))
        trades_df = trades_df.where(trades_df.notna(), None)
        trades_records = trades_df.to_dict(orient="records")
    else:
        trades_records = []

    metrics = {key: _to_python_number(value) for key, value in report.metrics.items()}

    payload = {
        "id": record_id,
        "created_at": datetime.utcnow().isoformat(),
        "context": context,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "price_data": price_records.to_dict(orient="records"),
        "trades": trades_records,
    }

    file_path = storage / f"{record_id}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return record_id


def list_backtests() -> List[Dict[str, Any]]:
    storage = _storage_dir()
    records: List[Dict[str, Any]] = []
    for file_path in sorted(storage.glob("*.json"), reverse=True):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        context = data.get("context", {})
        parameters = context.get("parameters", {})
        records.append(
            {
                "id": data.get("id", file_path.stem),
                "created_at": data.get("created_at"),
                "parameters": parameters,
                "metrics": data.get("metrics", {}),
                "settings": context.get("settings", {}),
            }
        )
    return records


def load_backtest(backtest_id: str) -> Optional[Dict[str, Any]]:
    storage = _storage_dir()
    file_path = storage / f"{backtest_id}.json"
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)
