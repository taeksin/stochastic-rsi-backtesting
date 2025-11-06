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

    equity_path = storage / f"{record_id}_equity.csv"
    trades_path = storage / f"{record_id}_trades.csv"

    price_path = storage / f"{record_id}_price.csv"
    price_records = getattr(report, "price_frame", price_df)
    price_records = price_records.copy()
    if not price_records.empty:
        if "timestamp" in price_records.columns:
            ts = pd.to_datetime(price_records["timestamp"], errors="coerce")
            if hasattr(ts.dt, "tz"):
                try:
                    ts = ts.dt.tz_localize(None)
                except TypeError:
                    ts = ts.dt.tz_convert(None)
            price_records["timestamp"] = ts
        price_records.to_csv(price_path, index=False, encoding="utf-8-sig")
    else:
        price_path.touch()

    equity_series = report.equity_curve
    if not isinstance(equity_series.index, pd.DatetimeIndex):
        equity_series.index = pd.to_datetime(equity_series.index, errors="coerce")
    equity_df = pd.DataFrame(
        {
            "timestamp": equity_series.index.astype(str),
            "equity": equity_series.astype(float).tolist(),
        }
    )
    equity_df = equity_df.sort_values("timestamp")
    equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")

    expected_columns = [
        "entry_index",
        "entry_time",
        "entry_price",
        "entry_ma_value",
        "exit_index",
        "exit_time",
        "direction",
        "pnl_pct",
        "pnl_value",
        "balance",
        "exit_reason",
        "trade_capital",
        "exit_signal",
        "exit_price",
        "exit_type",
        "take_profit_price",
        "stop_loss_price",
    ]

    trades_records: List[Dict[str, Any]] = []
    if report.trades is not None and not report.trades.empty:
        trades_df = report.trades.copy()
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
        trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        trades_records = []
    else:
        pd.DataFrame(columns=expected_columns).to_csv(trades_path, index=False, encoding="utf-8-sig")
        trades_records = []

    metrics = {key: _to_python_number(value) for key, value in report.metrics.items()}

    payload = {
        "id": record_id,
        "created_at": datetime.utcnow().isoformat(),
        "context": context,
        "metrics": metrics,
        "files": {
            "price_csv": price_path.name,
            "equity_curve_csv": equity_path.name,
            "trades_csv": trades_path.name,
        },
        "price_data": [],
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


def load_price_csv(metadata: Dict[str, Any]) -> pd.DataFrame:
    files = metadata.get("files", {})
    path = files.get("price_csv")
    storage = _storage_dir()
    if path:
        csv_path = storage / path
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
            if not df.empty and "exit_signal" in df.columns:
                df["exit_signal"] = df["exit_signal"].fillna("").astype(str)
            return df
    return pd.DataFrame(metadata.get("price_data", []))


def load_equity_curve_csv(metadata: Dict[str, Any]) -> pd.Series:
    files = metadata.get("files", {})
    path = files.get("equity_curve_csv")
    if path:
        storage = _storage_dir()
        csv_path = storage / path
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.sort_values("timestamp")
                return pd.Series(df["equity"].values, index=df["timestamp"])
            return pd.Series(df.iloc[:, 0].values)

    entries = metadata.get("equity_curve", [])
    if entries:
        index = pd.to_datetime([entry.get("timestamp") for entry in entries], errors="coerce")
        values = [entry.get("equity") for entry in entries]
        return pd.Series(values, index=index)
    return pd.Series(dtype=float)


def load_trades_csv(metadata: Dict[str, Any]) -> pd.DataFrame:
    files = metadata.get("files", {})
    path = files.get("trades_csv")
    storage = _storage_dir()
    df: pd.DataFrame
    if path:
        csv_path = storage / path
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame(metadata.get("trades", []))

    if not df.empty:
        if "exit_reason" in df.columns:
            df["exit_reason"] = df["exit_reason"].fillna("").astype(str)
        if "exit_signal" in df.columns:
            df["exit_signal"] = df["exit_signal"].fillna("").astype(str)
        if "exit_price" in df.columns:
            df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
        if "entry_price" in df.columns:
            df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        if "entry_ma_value" in df.columns:
            df["entry_ma_value"] = pd.to_numeric(df["entry_ma_value"], errors="coerce")
        if "exit_type" in df.columns:
            df["exit_type"] = df["exit_type"].fillna("").astype(str)
        if "take_profit_price" in df.columns:
            df["take_profit_price"] = pd.to_numeric(df["take_profit_price"], errors="coerce")
        if "stop_loss_price" in df.columns:
            df["stop_loss_price"] = pd.to_numeric(df["stop_loss_price"], errors="coerce")
        if "trade_capital" in df.columns:
            df["trade_capital"] = pd.to_numeric(df["trade_capital"], errors="coerce")
        if "pnl_value" in df.columns:
            df["pnl_value"] = pd.to_numeric(df["pnl_value"], errors="coerce")
        if "pnl_pct" in df.columns:
            df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    return df
