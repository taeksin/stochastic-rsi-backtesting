from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.backtesting import metrics
from src.strategy.base_strategy import BaseStrategy
from src.utils.validators import ensure_columns


@dataclass(slots=True)
class BacktestReport:
    equity_curve: pd.Series
    returns: pd.Series
    metrics: dict
    trades: pd.DataFrame
    price_frame: pd.DataFrame


class BacktestEngine:
    def __init__(self, fee_rate: float = 0.0004, default_trade_capital: Optional[float] = None):
        self.fee_rate = fee_rate
        self.default_trade_capital = default_trade_capital

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float,
        leverage: int,
        trade_capital: Optional[float] = None,
        take_profit_pct: float = 0.02,
        stop_loss_pct: float = 0.015,
    ) -> BacktestReport:
        ensure_columns(df, ["timestamp", "open", "high", "low", "close"])
        data = strategy.generate_signals(df)
        ensure_columns(
            data,
            ["timestamp", "open", "high", "low", "close", "signal", "position"],
        )

        if trade_capital is None:
            trade_capital = (
                self.default_trade_capital
                if self.default_trade_capital is not None
                else initial_capital
            )
        else:
            self.default_trade_capital = trade_capital

        capital_fraction = trade_capital / initial_capital if initial_capital else 0.0

        net_returns = np.zeros(len(data), dtype=float)
        trades: list[dict] = []

        signals = data["signal"].fillna(0).astype(int)
        target_positions = data["position"].fillna(0).astype(int)

        position = 0
        entry_price: Optional[float] = None
        entry_idx: Optional[int] = None
        entry_time: Optional[pd.Timestamp] = None
        entry_fee_return = 0.0

        def _to_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
            if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                return None
            if isinstance(ts, pd.Timestamp):
                if pd.isna(ts):
                    return None
                try:
                    ts = ts.tz_localize(None)
                except AttributeError:
                    pass
                except TypeError:
                    ts = ts.tz_convert(None)
                return ts.isoformat()
            return str(ts)

        rows = data.reset_index(drop=True)
        if "exit_signal" not in rows.columns:
            rows["exit_signal"] = ""
        if "cooldown_active" not in rows.columns:
            rows["cooldown_active"] = False

        for idx, row in rows.iterrows():
            ts = pd.to_datetime(row["timestamp"], errors="coerce")
            close_price = float(row["close"])
            high_price = float(row["high"])
            low_price = float(row["low"])
            signal = signals.iloc[idx]

            target = target_positions.iloc[idx]

            if position == 0:
                if signal in (1, -1):
                    position = signal
                    entry_price = close_price
                    entry_idx = idx
                    entry_time = ts
                    entry_fee_return = self.fee_rate * capital_fraction * leverage
                    net_returns[idx] -= entry_fee_return
                continue

            exit_flag = False
            exit_price = close_price
            exit_reason_code = "unknown_exit"
            context_row = rows.iloc[idx]

            if position == 1 and entry_price is not None:
                tp_price = entry_price * (1 + take_profit_pct)
                sl_price = entry_price * (1 - stop_loss_pct)
                if low_price <= sl_price:
                    exit_price = sl_price
                    exit_reason_code = "stop_loss_long"
                    exit_flag = True
                elif high_price >= tp_price:
                    exit_price = tp_price
                    exit_reason_code = "take_profit_long"
                    exit_flag = True
            elif position == -1 and entry_price is not None:
                tp_price = entry_price * (1 - take_profit_pct)
                sl_price = entry_price * (1 + stop_loss_pct)
                if high_price >= sl_price:
                    exit_price = sl_price
                    exit_reason_code = "stop_loss_short"
                    exit_flag = True
                elif low_price <= tp_price:
                    exit_price = tp_price
                    exit_reason_code = "take_profit_short"
                    exit_flag = True

            if not exit_flag and target == 0:
                exit_flag = True
                detail = str(context_row.get("exit_signal", ""))
                if detail == "dead_cross_exit":
                    exit_reason_code = "strategy_exit_dead_cross"
                elif detail == "golden_cross_exit":
                    exit_reason_code = "strategy_exit_golden_cross"
                elif detail == "cooldown_exit":
                    exit_reason_code = "strategy_exit_cooldown"
                elif detail == "bias_neutral_exit":
                    exit_reason_code = "strategy_exit_ma_neutral"
                else:
                    exit_reason_code = "strategy_exit_other"
                exit_price = close_price
            elif not exit_flag and target == -position:
                exit_flag = True
                exit_reason_code = "reverse_to_long" if target > position else "reverse_to_short"
                exit_price = close_price

            if exit_flag and entry_price is not None and entry_idx is not None:
                base_pct = (exit_price - entry_price) / entry_price
                pnl_pct = base_pct if position == 1 else -base_pct
                gross_return = pnl_pct * leverage * capital_fraction
                exit_fee_return = self.fee_rate * capital_fraction * leverage
                net_returns[idx] += gross_return - exit_fee_return
                total_net_return = gross_return - exit_fee_return - entry_fee_return
                pnl_value = total_net_return * initial_capital
                trade_pct_net = pnl_value / trade_capital if trade_capital else np.nan

                trades.append(
                    {
                        "entry_index": entry_idx,
                        "entry_time": _to_iso(entry_time),
                        "exit_index": idx,
                        "exit_time": _to_iso(ts),
                        "direction": "long" if position > 0 else "short",
                        "pnl_pct": trade_pct_net,
                        "pnl_value": pnl_value,
                        "trade_capital": trade_capital,
                        "exit_reason": exit_reason_code,
                        "exit_signal": str(context_row.get("exit_signal", "")),
                        "exit_price": exit_price,
                        "balance_index": idx,
                    }
                )

                position = 0
                entry_price = None
                entry_idx = None
                entry_time = None
                entry_fee_return = 0.0

                allow_reentry = exit_reason_code.startswith("reverse_") and signal in (1, -1)
                if allow_reentry:
                    position = signal
                    entry_price = close_price
                    entry_idx = idx
                    entry_time = ts
                    entry_fee_return = self.fee_rate * capital_fraction * leverage
                    net_returns[idx] -= entry_fee_return
                continue

        if position != 0 and entry_price is not None and entry_idx is not None:
            idx = len(rows) - 1
            exit_price = float(rows.iloc[-1]["close"])
            ts = pd.to_datetime(rows.iloc[-1]["timestamp"], errors="coerce")
            base_pct = (exit_price - entry_price) / entry_price
            pnl_pct = base_pct if position == 1 else -base_pct
            gross_return = pnl_pct * leverage * capital_fraction
            exit_fee_return = self.fee_rate * capital_fraction * leverage
            net_returns[idx] += gross_return - exit_fee_return
            total_net_return = gross_return - exit_fee_return - entry_fee_return
            pnl_value = total_net_return * initial_capital
            trade_pct_net = pnl_value / trade_capital if trade_capital else np.nan
            trades.append(
                {
                    "entry_index": entry_idx,
                    "entry_time": _to_iso(entry_time),
                    "exit_index": idx,
                    "exit_time": _to_iso(ts),
                    "direction": "long" if position > 0 else "short",
                    "pnl_pct": trade_pct_net,
                    "pnl_value": pnl_value,
                    "trade_capital": trade_capital,
                    "exit_reason": "end_of_data",
                    "exit_signal": str(rows.iloc[idx].get("exit_signal", "")),
                    "exit_price": exit_price,
                    "balance_index": idx,
                }
            )

        net_returns_series = pd.Series(net_returns, index=data["timestamp"])
        equity_curve = (1 + net_returns_series).cumprod() * initial_capital

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            balances = []
            for b_idx in trades_df["balance_index"]:
                b_idx = min(int(b_idx), len(equity_curve) - 1)
                balances.append(float(equity_curve.iloc[b_idx]))
            trades_df["balance"] = balances
            trades_df.drop(columns=["balance_index"], inplace=True)
        pnl_series = trades_df["pnl_value"] if not trades_df.empty else pd.Series(dtype=float)

        stats = {
            "total_return": (equity_curve.iloc[-1] / initial_capital) - 1
            if not equity_curve.empty
            else np.nan,
            "sharpe_ratio": metrics.calculate_sharpe(net_returns_series),
            "max_drawdown": metrics.calculate_max_drawdown(equity_curve),
            "win_rate": metrics.calculate_win_rate(pnl_series)
            if not pnl_series.empty
            else np.nan,
            "trades": len(trades_df) if not trades_df.empty else 0,
            "final_equity": equity_curve.iloc[-1] if not equity_curve.empty else initial_capital,
            "trade_capital": trade_capital,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
            "initial_capital": initial_capital,
        }

        report = BacktestReport(
            equity_curve=equity_curve,
            returns=net_returns_series,
            metrics=stats,
            trades=trades_df,
            price_frame=rows,
        )
        return report
