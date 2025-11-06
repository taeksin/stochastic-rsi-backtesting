from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.backtesting import metrics
from src.strategy.base_strategy import BaseStrategy
from src.utils.validators import ensure_columns


@dataclass
class BacktestReport:
    equity_curve: pd.Series
    returns: pd.Series
    metrics: dict
    trades: pd.DataFrame
    price_frame: pd.DataFrame


class BacktestEngine:
    @dataclass
    class PositionState:
        direction: int
        entry_price: float
        entry_idx: int
        entry_time: Optional[pd.Timestamp]
        take_profit_price: float
        stop_loss_price: float
        entry_fee_return: float
        entry_ma_value: Optional[float]
        entry_rsi_value: Optional[float]

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
        take_profit_pct = take_profit_pct / 100 if take_profit_pct > 1 else take_profit_pct
        stop_loss_pct = stop_loss_pct / 100 if stop_loss_pct > 1 else stop_loss_pct
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

        rows = data.reset_index(drop=True).copy()
        if "exit_signal" not in rows.columns:
            rows["exit_signal"] = ""
        if "cooldown_active" not in rows.columns:
            rows["cooldown_active"] = False
        rows["engine_position"] = 0
        rows["engine_entry_price"] = np.nan
        rows["engine_take_profit"] = np.nan
        rows["engine_stop_loss"] = np.nan
        rows["engine_exit_type"] = ""
        rows["engine_exit_reason"] = ""
        rows["engine_exit_price"] = np.nan
        rows["engine_entry_ma"] = np.nan
        rows["engine_entry_rsi"] = np.nan

        def _set_row_position_state(idx: int, state: Optional[BacktestEngine.PositionState]) -> None:
            if state is None:
                rows.at[idx, "engine_position"] = 0
                rows.at[idx, "engine_entry_price"] = np.nan
                rows.at[idx, "engine_take_profit"] = np.nan
                rows.at[idx, "engine_stop_loss"] = np.nan
                rows.at[idx, "engine_entry_ma"] = np.nan
                rows.at[idx, "engine_entry_rsi"] = np.nan
            else:
                rows.at[idx, "engine_position"] = state.direction
                rows.at[idx, "engine_entry_price"] = state.entry_price
                rows.at[idx, "engine_take_profit"] = state.take_profit_price
                rows.at[idx, "engine_stop_loss"] = state.stop_loss_price
                rows.at[idx, "engine_entry_ma"] = state.entry_ma_value
                rows.at[idx, "engine_entry_rsi"] = state.entry_rsi_value

        def _calculate_targets(entry_px: float, direction: int) -> tuple[float, float]:
            if direction > 0:
                tp_price = entry_px * (1 + take_profit_pct)
                sl_price = entry_px * (1 - stop_loss_pct)
            else:
                tp_price = entry_px * (1 - take_profit_pct)
                sl_price = entry_px * (1 + stop_loss_pct)
            return tp_price, sl_price

        def _open_position(
            direction: int,
            price: float,
            idx: int,
            ts: Optional[pd.Timestamp],
            ma_value: Optional[float],
            rsi_value: Optional[float],
        ) -> BacktestEngine.PositionState:
            take_profit_price, stop_loss_price = _calculate_targets(price, direction)
            entry_fee_return = self.fee_rate * capital_fraction * leverage
            net_returns[idx] -= entry_fee_return
            state = BacktestEngine.PositionState(
                direction=direction,
                entry_price=price,
                entry_idx=idx,
                entry_time=ts,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                entry_fee_return=entry_fee_return,
                entry_ma_value=ma_value,
                entry_rsi_value=float(rsi_value) if rsi_value is not None and not pd.isna(rsi_value) else None,
            )
            _set_row_position_state(idx, state)
            return state

        position_state: Optional[BacktestEngine.PositionState] = None
        last_valid_rsi: Optional[float] = None

        for idx, row in rows.iterrows():
            ts = pd.to_datetime(row["timestamp"], errors="coerce")
            close_price = float(row["close"])
            high_price = float(row["high"])
            low_price = float(row["low"])
            signal = signals.iloc[idx]
            target = target_positions.iloc[idx]

            context_row = rows.iloc[idx]
            prior_exit_signal = str(context_row.get("exit_signal", ""))

            ma_value = row.get("ma_trend")
            rsi_value = row.get("rsi")
            if rsi_value is not None and not pd.isna(rsi_value):
                last_valid_rsi = float(rsi_value)
            else:
                rsi_value = last_valid_rsi
            if position_state is None:
                _set_row_position_state(idx, None)
                if signal in (1, -1):
                    position_state = _open_position(signal, close_price, idx, ts, ma_value, rsi_value)
                continue

            exit_flag = False
            exit_price = close_price
            exit_reason_code = "unknown_exit"
            exit_type = "unknown"
            exit_signal_value = prior_exit_signal

            direction = position_state.direction
            direction_label = "long" if direction > 0 else "short"
            _set_row_position_state(idx, position_state)

            if direction > 0:
                if low_price <= position_state.stop_loss_price:
                    exit_price = position_state.stop_loss_price
                    exit_reason_code = f"stop_loss_{direction_label}"
                    exit_signal_value = exit_reason_code
                    exit_type = "stop_loss"
                    exit_flag = True
                elif high_price >= position_state.take_profit_price:
                    exit_price = position_state.take_profit_price
                    exit_reason_code = f"take_profit_{direction_label}"
                    exit_signal_value = exit_reason_code
                    exit_type = "take_profit"
                    exit_flag = True
            else:
                if high_price >= position_state.stop_loss_price:
                    exit_price = position_state.stop_loss_price
                    exit_reason_code = f"stop_loss_{direction_label}"
                    exit_signal_value = exit_reason_code
                    exit_type = "stop_loss"
                    exit_flag = True
                elif low_price <= position_state.take_profit_price:
                    exit_price = position_state.take_profit_price
                    exit_reason_code = f"take_profit_{direction_label}"
                    exit_signal_value = exit_reason_code
                    exit_type = "take_profit"
                    exit_flag = True

            if not exit_flag and target == 0:
                exit_flag = True
                detail = str(context_row.get("exit_signal", ""))
                if detail in {"ma_bias_flip_to_long", "ma_bias_flip_to_short"}:
                    exit_reason_code = detail
                    exit_type = "ma_bias_flip"
                elif detail == "ma_bias_neutral_exit":
                    exit_reason_code = detail
                    exit_type = "ma_bias_flip"
                elif detail == "dead_cross_exit":
                    exit_reason_code = "strategy_exit_dead_cross"
                    exit_type = "strategy_exit"
                elif detail == "golden_cross_exit":
                    exit_reason_code = "strategy_exit_golden_cross"
                    exit_type = "strategy_exit"
                elif detail == "cooldown_exit":
                    exit_reason_code = "strategy_exit_cooldown"
                    exit_type = "strategy_exit"
                elif detail == "bias_neutral_exit":
                    exit_reason_code = "strategy_exit_ma_neutral"
                    exit_type = "strategy_exit"
                else:
                    exit_reason_code = "strategy_exit_other"
                    exit_type = "strategy_exit"
                exit_signal_value = detail or exit_reason_code
                exit_price = close_price
            elif not exit_flag and target == -direction:
                exit_flag = True
                exit_reason_code = "reverse_to_long" if target > direction else "reverse_to_short"
                exit_signal_value = exit_reason_code
                exit_price = close_price
                exit_type = "reverse"

            if exit_flag:
                base_pct = (exit_price - position_state.entry_price) / position_state.entry_price
                pnl_pct = base_pct if direction > 0 else -base_pct
                gross_return = pnl_pct * leverage * capital_fraction
                exit_fee_return = self.fee_rate * capital_fraction * leverage
                net_returns[idx] += gross_return - exit_fee_return
                total_net_return = gross_return - exit_fee_return - position_state.entry_fee_return
                pnl_value = total_net_return * initial_capital
                trade_pct_net = pnl_value / trade_capital if trade_capital else np.nan

                trades.append(
                    {
                        "entry_index": position_state.entry_idx,
                        "entry_time": _to_iso(position_state.entry_time),
                        "entry_price": position_state.entry_price,
                        "entry_ma_value": position_state.entry_ma_value,
                        "entry_rsi": position_state.entry_rsi_value,
                        "exit_index": idx,
                        "exit_time": _to_iso(ts),
                        "direction": "long" if direction > 0 else "short",
                        "pnl_pct": trade_pct_net,
                        "pnl_value": pnl_value,
                        "trade_capital": trade_capital,
                        "exit_reason": exit_reason_code,
                        "exit_signal": exit_signal_value,
                        "exit_price": exit_price,
                        "exit_type": exit_type,
                        "take_profit_price": position_state.take_profit_price,
                        "stop_loss_price": position_state.stop_loss_price,
                        "balance_index": idx,
                    }
                )

                rows.at[idx, "engine_exit_type"] = exit_type
                rows.at[idx, "engine_exit_reason"] = exit_reason_code
                rows.at[idx, "engine_exit_price"] = exit_price

                allow_reentry = exit_type == "reverse" and signal in (1, -1)
                position_state = None

                if allow_reentry:
                    position_state = _open_position(signal, close_price, idx, ts, ma_value, rsi_value)

        if position_state is not None:
            idx = len(rows) - 1
            exit_price = float(rows.iloc[-1]["close"])
            ts = pd.to_datetime(rows.iloc[-1]["timestamp"], errors="coerce")
            direction = position_state.direction
            base_pct = (exit_price - position_state.entry_price) / position_state.entry_price
            pnl_pct = base_pct if direction > 0 else -base_pct
            gross_return = pnl_pct * leverage * capital_fraction
            exit_fee_return = self.fee_rate * capital_fraction * leverage
            net_returns[idx] += gross_return - exit_fee_return
            total_net_return = gross_return - exit_fee_return - position_state.entry_fee_return
            pnl_value = total_net_return * initial_capital
            trade_pct_net = pnl_value / trade_capital if trade_capital else np.nan
            trades.append(
                {
                    "entry_index": position_state.entry_idx,
                    "entry_time": _to_iso(position_state.entry_time),
                    "entry_price": position_state.entry_price,
                    "entry_ma_value": position_state.entry_ma_value,
                    "entry_rsi": position_state.entry_rsi_value,
                    "exit_index": idx,
                    "exit_time": _to_iso(ts),
                    "direction": "long" if direction > 0 else "short",
                    "pnl_pct": trade_pct_net,
                    "pnl_value": pnl_value,
                    "trade_capital": trade_capital,
                    "exit_reason": "end_of_data",
                    "exit_signal": "end_of_data",
                    "exit_price": exit_price,
                    "exit_type": "end_of_data",
                    "take_profit_price": position_state.take_profit_price,
                    "stop_loss_price": position_state.stop_loss_price,
                    "balance_index": idx,
                }
            )
            rows.at[idx, "engine_exit_type"] = "end_of_data"
            rows.at[idx, "engine_exit_reason"] = "end_of_data"
            rows.at[idx, "engine_exit_price"] = exit_price
            _set_row_position_state(idx, None)

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
