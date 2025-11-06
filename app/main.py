from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import io
import textwrap
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.components import charts, metrics_display, sidebar
from config.settings import settings
from src.backtesting.engine import BacktestEngine
from src.data.bybit_client import fetch_bybit_klines
from src.data.resampler import KlineResampler
from src.strategy.ma200_stochrsi import MA200StochRSIStrategy, StrategyParameters
from src.utils.logger_confg import get_logger
from src.utils import storage
from src.utils.excel_styles import style_trade_sheet

logger = get_logger()


SUPPORTED_INTERVALS = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "1440"}


def _timeframe_to_minutes(value: str) -> int:
    upper = value.strip().upper()
    if upper.endswith("T"):
        return int(upper[:-1])
    if upper.endswith("H"):
        return int(float(upper[:-1]) * 60)
    if upper.endswith("D"):
        return int(float(upper[:-1]) * 60 * 24)
    return int(upper)


def _select_fetch_interval(minutes: int) -> str:
    if minutes <= 0:
        return "1"
    supported = sorted(int(item) for item in SUPPORTED_INTERVALS)
    if minutes in supported:
        return str(minutes)
    divisors = [value for value in supported if value <= minutes and minutes % value == 0]
    if divisors:
        return str(max(divisors))
    return "1"


def load_price_data(
    start: datetime,
    end: datetime,
    timeframe: str,
    *,
    symbol: str = "BTCUSDT",
    category: str = "linear",
) -> pd.DataFrame:
    minutes = _timeframe_to_minutes(timeframe)
    interval = str(minutes)
    fetch_interval = _select_fetch_interval(minutes)

    progress_placeholder = None
    progress_bar = None
    try:
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0.0)
    except Exception:  # Streamlit context unavailable
        progress_placeholder = None
        progress_bar = None

    def _update_progress(completed: int, total: int) -> None:
        if progress_bar is None or total <= 0:
            return
        fraction = min(completed / total, 1.0)
        progress_bar.progress(fraction)

    try:
        fetched = fetch_bybit_klines(
            start=start,
            end=end,
            interval_minutes=fetch_interval,
            symbol=symbol,
            category=category,
            show_progress=False,
            sequential=True,
            progress_hook=_update_progress,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to fetch Bybit price data", exc_info=exc)
        if progress_placeholder is not None:
            progress_placeholder.empty()
        return pd.DataFrame()

    if fetched.empty:
        if progress_placeholder is not None:
            progress_placeholder.empty()
        return fetched

    price_df = fetched.copy()
    if fetch_interval != interval:
        try:
            resampled = KlineResampler.resample_to_timeframe(price_df, timeframe.upper())
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to resample fetched data; falling back to raw resolution", exc_info=exc)
            resampled = price_df.copy()
        price_df = resampled

    price_df = price_df.drop(
        columns=[col for col in price_df.columns if col not in {"timestamp", "open", "high", "low", "close", "volume"}],
        errors="ignore",
    )
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], errors="coerce")
    price_df["timestamp"] = price_df["timestamp"].dt.tz_localize(None)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in price_df.columns:
            price_df[column] = pd.to_numeric(price_df[column], errors="coerce")
    price_df = price_df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if progress_placeholder is not None:
        progress_placeholder.empty()
    return price_df.sort_values("timestamp").reset_index(drop=True)




def format_trade_table(
    trades_df: pd.DataFrame,
    take_profit_pct: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    *,
    return_export: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if take_profit_pct is not None and take_profit_pct > 1:
        take_profit_pct = take_profit_pct / 100
    if stop_loss_pct is not None and stop_loss_pct > 1:
        stop_loss_pct = stop_loss_pct / 100

    columns = [
        "진입 시각",
        "청산 시각",
        "포지션",
        "청산 유형",
        "수익률(%)",
        "수익 (₩)",
        "투입 자본 (₩)",
        "진입가 (₩)",
        "진입 MA 값",
        "진입 RSI",
        "청산가 (₩)",
        "손절 기준 (₩)",
        "익절 기준 (₩)",
        "잔액 (₩)",
        "종료 사유",
    ]

    if trades_df is None or trades_df.empty:
        empty_df = pd.DataFrame(columns=columns)
        if return_export:
            return empty_df, empty_df.copy()
        return empty_df

    df = trades_df.copy()

    entry_time = (
        pd.to_datetime(df.get("entry_time"), errors="coerce")
        if "entry_time" in df
        else pd.Series(pd.NaT, index=df.index)
    ).dt.tz_localize(None)
    exit_time = (
        pd.to_datetime(df.get("exit_time"), errors="coerce")
        if "exit_time" in df
        else pd.Series(pd.NaT, index=df.index)
    ).dt.tz_localize(None)

    raw_direction = df.get("direction", pd.Series("", index=df.index)).fillna("").astype(str)
    direction = raw_direction.map({"long": "롱", "short": "숏"}).fillna(raw_direction)
    lower_direction = raw_direction.str.lower()

    pnl_pct = pd.to_numeric(df.get("pnl_pct"), errors="coerce") if "pnl_pct" in df else pd.Series(0.0, index=df.index)
    pnl_value = pd.to_numeric(df.get("pnl_value"), errors="coerce") if "pnl_value" in df else pd.Series(0.0, index=df.index)
    capital_used = pd.to_numeric(df.get("trade_capital"), errors="coerce") if "trade_capital" in df else pd.Series(0.0, index=df.index)
    balance = pd.to_numeric(df.get("balance"), errors="coerce") if "balance" in df else pd.Series(0.0, index=df.index)
    exit_price = pd.to_numeric(df.get("exit_price"), errors="coerce") if "exit_price" in df else pd.Series(np.nan, index=df.index)
    entry_price = pd.to_numeric(df.get("entry_price"), errors="coerce") if "entry_price" in df else pd.Series(np.nan, index=df.index)
    entry_ma_value = pd.to_numeric(df.get("entry_ma_value"), errors="coerce") if "entry_ma_value" in df else pd.Series(np.nan, index=df.index)
    entry_rsi = pd.to_numeric(df.get("entry_rsi"), errors="coerce") if "entry_rsi" in df else pd.Series(np.nan, index=df.index)
    reason_codes = df.get("exit_reason", pd.Series("", index=df.index)).fillna("").astype(str)
    signal_codes = df.get("exit_signal", pd.Series("", index=df.index)).fillna("").astype(str)
    exit_types = df.get("exit_type", pd.Series("", index=df.index)).fillna("").astype(str)
    take_profit_target = pd.to_numeric(df.get("take_profit_price"), errors="coerce") if "take_profit_price" in df else pd.Series(np.nan, index=df.index)
    stop_loss_target = pd.to_numeric(df.get("stop_loss_price"), errors="coerce") if "stop_loss_price" in df else pd.Series(np.nan, index=df.index)

    if take_profit_pct is not None and not pd.isna(take_profit_pct) and take_profit_target.isna().any():
        tp_fallback = np.where(
            lower_direction == "short",
            entry_price * (1 - take_profit_pct),
            entry_price * (1 + take_profit_pct),
        )
        take_profit_target = take_profit_target.where(take_profit_target.notna(), pd.Series(tp_fallback, index=df.index))

    if stop_loss_pct is not None and not pd.isna(stop_loss_pct) and stop_loss_target.isna().any():
        sl_fallback = np.where(
            lower_direction == "short",
            entry_price * (1 + stop_loss_pct),
            entry_price * (1 - stop_loss_pct),
        )
        stop_loss_target = stop_loss_target.where(stop_loss_target.notna(), pd.Series(sl_fallback, index=df.index))

    reason_labels = {
        "strategy_exit_dead_cross": "스토캐스틱 데드크로스 재발생",
        "strategy_exit_golden_cross": "스토캐스틱 골든크로스 재발생",
        "strategy_exit_cooldown": "MA 쿨다운 진행 중",
        "strategy_exit_ma_neutral": "MA 바이어스 중립화",
        "ma_bias_flip_to_long": "MA 모드가 롱으로 전환",
        "ma_bias_flip_to_short": "MA 모드가 숏으로 전환",
        "ma_bias_neutral_exit": "MA 모드가 중립으로 전환",
        "reverse_to_long": "숏 → 롱 전환",
        "reverse_to_short": "롱 → 숏 전환",
        "strategy_exit_other": "전략 종료 (기타)",
        "take_profit_long": "롱 포지션 익절",
        "take_profit_short": "숏 포지션 익절",
        "stop_loss_long": "롱 포지션 손절",
        "stop_loss_short": "숏 포지션 손절",
    }
    signal_labels = {
        "dead_cross_exit": "스토캐스틱 데드크로스",
        "golden_cross_exit": "스토캐스틱 골든크로스",
        "cooldown_exit": "MA 쿨다운",
        "bias_neutral_exit": "MA 바이어스 중립",
    }
    exit_type_labels = {
        "take_profit": "익절",
        "stop_loss": "손절",
        "strategy_exit": "전략 종료",
        "reverse": "포지션 전환",
        "end_of_data": "데이터 종료",
        "ma_bias_flip": "MA 모드 전환",
    }

    def format_price(value: float) -> str:
        return f"{value:,.0f}" if pd.notna(value) else "N/A"

    exit_type_display: list[str] = []
    detailed_reason: list[str] = []
    for idx, exit_type_value in exit_types.items():
        code = reason_codes.iloc[idx]
        signal = signal_codes.iloc[idx]
        price = exit_price.iloc[idx]
        exit_label = exit_type_labels.get(exit_type_value, "기타")
        exit_type_display.append(exit_label)

        if exit_type_value == "take_profit":
            desc = reason_labels.get(code, "목표가 도달")
            detailed_reason.append(f"{desc} (가격 {format_price(price)})")
        elif exit_type_value == "stop_loss":
            desc = reason_labels.get(code, "손절가 도달")
            detailed_reason.append(f"{desc} (가격 {format_price(price)})")
        elif exit_type_value == "reverse":
            desc = reason_labels.get(code, "반대 포지션 전환")
            detailed_reason.append(f"{desc} (가격 {format_price(price)})")
        elif exit_type_value in {"strategy_exit", "ma_bias_flip"}:
            desc = reason_labels.get(code)
            if not desc or desc == "전략 종료 (기타)":
                desc = signal_labels.get(signal, desc or "전략 종료")
            detailed_reason.append(f"{desc} (가격 {format_price(price)})")
        elif exit_type_value == "end_of_data":
            detailed_reason.append(f"데이터 종료 시점 도달 (가격 {format_price(price)})")
        else:
            fallback = reason_labels.get(code, code if code else "-")
            detailed_reason.append(f"{fallback} (가격 {format_price(price)})")

    reason_series = pd.Series(detailed_reason, index=df.index)

    export_df = pd.DataFrame(
        {
            "진입 시각": entry_time,
            "청산 시각": exit_time,
            "포지션": direction,
            "청산 유형": exit_type_display,
            "수익률(%)": pnl_pct,
            "수익 (₩)": pnl_value,
            "투입 자본 (₩)": capital_used,
            "진입가 (₩)": entry_price,
            "진입 MA 값": entry_ma_value,
            "진입 RSI": entry_rsi,
            "청산가 (₩)": exit_price,
            "손절 기준 (₩)": stop_loss_target,
            "익절 기준 (₩)": take_profit_target,
            "잔액 (₩)": balance,
            "종료 사유": reason_series,
        }
    )[columns]

    formatted = export_df.copy()
    formatted["진입 시각"] = formatted["진입 시각"].dt.strftime("%Y-%m-%d %H:%M")
    formatted["청산 시각"] = formatted["청산 시각"].dt.strftime("%Y-%m-%d %H:%M")
    formatted["수익률(%)"] = formatted["수익률(%)"].map(lambda x: f"{x * 100:.2f}" if pd.notna(x) else "N/A")

    money_columns = [
        "수익 (₩)",
        "투입 자본 (₩)",
        "진입가 (₩)",
        "진입 MA 값",
        "청산가 (₩)",
        "손절 기준 (₩)",
        "익절 기준 (₩)",
        "잔액 (₩)",
    ]
    for col in money_columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")

    formatted["진입 RSI"] = formatted["진입 RSI"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    formatted.index = formatted.index + 1

    if return_export:
        return formatted, export_df
    return formatted


def _build_summary_sheet(
    context: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    initial_capital: float,
    position_capital: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    ma_cooldown_minutes: int,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    def _format_percent(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return f"{value * 100:.2f}%"

    def _format_number(value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return f"{value:,.0f}"

    def _format_datetime_str(value: Optional[str | pd.Timestamp]) -> str:
        if value is None:
            return "-"
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return "-"
            return value.strftime("%Y-%m-%d %H:%M")
        return str(value)

    params = context.get("parameters", {}) if context else {}
    settings_ctx = context.get("settings", {}) if context else {}
    strategy_ctx = context.get("strategy", {}) if context else {}

    timeframe_value = params.get("timeframe") or settings_ctx.get("timeframe")
    if not timeframe_value:
        timeframe_value = "-"
    start_value = params.get("start_date")
    end_value = params.get("end_date")
    if start_value is None and "timestamp" in price_df.columns and not price_df.empty:
        start_value = pd.to_datetime(price_df["timestamp"], errors="coerce").dropna().min()
    if end_value is None and "timestamp" in price_df.columns and not price_df.empty:
        end_value = pd.to_datetime(price_df["timestamp"], errors="coerce").dropna().max()

    rows: list[dict[str, str]] = []
    rows.extend(
        [
            {"구분": "설정", "항목": "시작일", "값": _format_datetime_str(start_value)},
            {"구분": "설정", "항목": "종료일", "값": _format_datetime_str(end_value)},
            {"구분": "설정", "항목": "타임프레임", "값": str(timeframe_value)},
            {"구분": "설정", "항목": "초기 자본 (₩)", "값": _format_number(initial_capital)},
            {"구분": "설정", "항목": "거래당 투입 자본 (₩)", "값": _format_number(position_capital)},
            {"구분": "설정", "항목": "레버리지", "값": f"x{settings_ctx.get('leverage', '-') or '-'}"},
            {"구분": "설정", "항목": "익절 기준 (%)", "값": _format_percent(take_profit_pct)},
            {"구분": "설정", "항목": "손절 기준 (%)", "값": _format_percent(stop_loss_pct)},
            {"구분": "설정", "항목": "MA 쿨다운 (분)", "값": str(ma_cooldown_minutes)},
            {"구분": "설정", "항목": "MA 기간", "값": str(strategy_ctx.get("ma_period", "-"))},
            {"구분": "설정", "항목": "RSI 기간", "값": str(strategy_ctx.get("rsi_period", "-"))},
            {"구분": "설정", "항목": "Stochastic 기간", "값": str(strategy_ctx.get("stoch_period", "-"))},
            {"구분": "설정", "항목": "Stoch K", "값": str(strategy_ctx.get("stoch_k", "-"))},
            {"구분": "설정", "항목": "Stoch D", "값": str(strategy_ctx.get("stoch_d", "-"))},
        ]
    )

    total_return = metrics.get("total_return")
    max_drawdown = metrics.get("max_drawdown")
    win_rate = metrics.get("win_rate")
    sharpe_ratio = metrics.get("sharpe_ratio")
    final_equity = metrics.get("final_equity")
    trades_count = metrics.get("trades")

    rows.extend(
        [
            {"구분": "결과", "항목": "총 수익률 (%)", "값": _format_percent(total_return)},
            {"구분": "결과", "항목": "최대 낙폭 (%)", "값": _format_percent(max_drawdown)},
            {"구분": "결과", "항목": "승률 (%)", "값": _format_percent(win_rate)},
            {"구분": "결과", "항목": "샤프 지수", "값": "-" if sharpe_ratio is None or (isinstance(sharpe_ratio, float) and np.isnan(sharpe_ratio)) else f"{sharpe_ratio:.2f}"},
            {"구분": "결과", "항목": "총 거래 수", "값": str(trades_count if trades_count is not None else "-")},
            {"구분": "결과", "항목": "최종 자산 (₩)", "값": _format_number(final_equity)},
        ]
    )

    return pd.DataFrame(rows, columns=["구분", "항목", "값"])


def render_backtest_results(
    price_df: pd.DataFrame,
    equity_curve: pd.Series,
    metrics: Dict[str, Any],
    trades_df: pd.DataFrame,
    initial_capital: float,
    position_capital: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    ma_cooldown_minutes: int,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    charts.render_price_chart(price_df)
    charts.render_equity_curve(equity_curve)
    metrics_display.render_metrics(metrics)

    st.subheader("거래 내역")
    st.caption(
        f"초기 자본: {initial_capital:,.0f} | 거래당 투입 자본: {position_capital:,.0f} | "
        f"손절: {stop_loss_pct * 100:.2f}% | 익절: {take_profit_pct * 100:.2f}% | "
        f"MA 쿨다운: {ma_cooldown_minutes}분"
    )

    display_df, export_df = format_trade_table(
        trades_df,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        return_export=True,
    )
    if display_df.empty:
        st.info("거래 내역이 없습니다.")
    else:
        max_rows = 500
        total_rows = len(display_df)
        truncated_df = display_df.head(max_rows)
        if total_rows > max_rows:
            st.info(f"총 {total_rows}건 중 처음 {max_rows}건만 표에 표시합니다. 전체 내역은 아래에서 다운로드하세요.")
        st.dataframe(truncated_df, use_container_width=True, height=420)

        excel_buffer = io.BytesIO()
        export_df_with_index = export_df.copy()
        export_df_with_index.insert(0, "No.", np.arange(1, len(export_df_with_index) + 1))

        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            summary_df = _build_summary_sheet(context, metrics, initial_capital, position_capital, take_profit_pct, stop_loss_pct, ma_cooldown_minutes, price_df)
            summary_df.to_excel(writer, index=False, sheet_name="summary")
            summary_ws = writer.sheets["summary"]
            summary_ws.set_column(0, 0, 12)
            summary_ws.set_column(1, 1, 22)
            summary_ws.set_column(2, 2, 28)

            export_df_with_index.to_excel(writer, index=False, sheet_name="trades")
            trades_ws = writer.sheets["trades"]
            trades_ws.freeze_panes(1, 0)
            style_trade_sheet(writer.book, trades_ws, export_df_with_index)
        excel_buffer.seek(0)
        st.download_button(
            "전체 거래 XLSX 다운로드",
            data=excel_buffer,
            file_name="trades.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )




def build_context(state: sidebar.SidebarState) -> Dict[str, Any]:
    return {
        "parameters": {
            "start_date": state.start_date.isoformat(),
            "end_date": state.end_date.isoformat(),
            "timeframe": state.timeframe,
        },
        "strategy": {
            "ma_period": state.ma_period,
            "rsi_period": state.rsi_period,
            "stoch_period": state.stoch_period,
            "stoch_k": state.stoch_k,
            "stoch_d": state.stoch_d,
        },
        "settings": {
            "initial_capital": settings.backtest.initial_capital,
            "position_capital": state.position_capital,
            "leverage": state.leverage,
            "take_profit_pct": state.take_profit_pct,
            "stop_loss_pct": state.stop_loss_pct,
            "ma_cooldown_minutes": state.ma_cooldown_minutes,
        },
    }


def load_saved_result(backtest_id: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Loading saved backtest: {backtest_id}")
    data = storage.load_backtest(backtest_id)
    if not data:
        logger.warning(f"Backtest record not found: {backtest_id}")
        return None

    price_loader = getattr(storage, "load_price_csv", None)
    if callable(price_loader):
        price_df = price_loader(data)
    else:
        price_df = pd.DataFrame(data.get("price_data", []))
    if not price_df.empty and "timestamp" in price_df.columns:
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], errors="coerce")
        if hasattr(price_df["timestamp"].dt, "tz"):
            try:
                price_df["timestamp"] = price_df["timestamp"].dt.tz_localize(None)
            except TypeError:
                pass
        price_df = price_df.sort_values("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in price_df.columns:
                price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
        for col in ["signal", "position"]:
            if col in price_df.columns:
                price_df[col] = pd.to_numeric(price_df[col], errors="coerce").fillna(0).astype(int)
        if "exit_signal" in price_df.columns:
            price_df["exit_signal"] = price_df["exit_signal"].fillna("").astype(str)

    equity_curve = storage.load_equity_curve_csv(data)
    trades_df = storage.load_trades_csv(data)

    result = {
        "price_df": price_df,
        "equity_curve": equity_curve,
        "metrics": data.get("metrics", {}),
        "trades": trades_df,
        "context": data.get("context", {}),
    }
    logger.info(f"Finished loading backtest: {backtest_id}")
    return result


def main() -> None:
    logger.info("Streamlit app initialization")
    st.set_page_config(page_title="Stochastic RSI Backtester", layout="wide")
    st.title("Stochastic RSI 백테스트")

    initial_capital = settings.backtest.initial_capital
    default_position_capital = getattr(
        settings.backtest,
        "position_capital",
        settings.backtest.initial_capital,
    )
    default_take_profit_pct = getattr(settings.backtest, "take_profit_pct", 0.02)
    default_stop_loss_pct = getattr(settings.backtest, "stop_loss_pct", 0.015)

    saved_backtests = storage.list_backtests()

    state = sidebar.render_sidebar(
        default_timeframe=settings.backtest.timeframe,
        default_leverage=settings.backtest.leverage,
        initial_capital=initial_capital,
        default_position_capital=default_position_capital,
        default_take_profit_pct=default_take_profit_pct,
        default_stop_loss_pct=default_stop_loss_pct,
        default_ma_cooldown=settings.backtest.ma_cooldown_minutes,
        saved_results=saved_backtests,
    )



    if state.mode == "분석":
        if not state.selected_backtest_id:
            st.info("사이드바에서 분석할 테스트를 선택해주세요.")
            return

        saved = load_saved_result(state.selected_backtest_id)
        if saved is None:
            st.error("선택한 테스트 기록을 불러오지 못했습니다.")
            return

        context = saved.get("context", {})
        parameters = context.get("parameters", {})
        settings_ctx = context.get("settings", {})
        tp_ctx = settings_ctx.get("take_profit_pct", default_take_profit_pct)
        sl_ctx = settings_ctx.get("stop_loss_pct", default_stop_loss_pct)
        cooldown_ctx = settings_ctx.get("ma_cooldown_minutes", settings.backtest.ma_cooldown_minutes)

        st.subheader("저장된 테스트 결과")
        summary_lines: list[str] = []
        if parameters:
            period = f"{parameters.get('start_date','')} ~ {parameters.get('end_date','')}"
            summary_lines.append(f"- **기간:** {period}")
            summary_lines.append(f"- **타임프레임:** {parameters.get('timeframe','')}")
        if context.get("strategy"):
            strategy = context["strategy"]
            summary_lines.append(
                "- **전략 파라미터:** "
                f"MA {strategy.get('ma_period')}, RSI {strategy.get('rsi_period')}, "
                f"Stoch {strategy.get('stoch_period')} (K={strategy.get('stoch_k')}, D={strategy.get('stoch_d')})"
            )
        summary_lines.append(
            "- **리스크 설정:** "
            f"TP {tp_ctx * 100:.2f}% / SL {sl_ctx * 100:.2f}% / 쿨다운 {cooldown_ctx}분"
        )
        if summary_lines:
            st.markdown("\n".join(summary_lines))

        initial_cap = settings_ctx.get("initial_capital", initial_capital)
        position_cap = settings_ctx.get("position_capital", default_position_capital)

        render_backtest_results(
            price_df=saved["price_df"],
            equity_curve=saved["equity_curve"],
            metrics=saved["metrics"],
            trades_df=saved["trades"],
            initial_capital=initial_cap,
            position_capital=position_cap,
            take_profit_pct=tp_ctx,
            stop_loss_pct=sl_ctx,
            ma_cooldown_minutes=cooldown_ctx,
            context=context,
        )
        return



    data_start = datetime.combine(state.start_date, datetime.min.time())
    data_end = datetime.combine(state.end_date, datetime.max.time())

    if not state.run_backtest:
        st.info("사이드바에서 파라미터를 설정한 뒤 '백테스트 실행' 버튼을 눌러주세요.")
        st.markdown(
            textwrap.dedent(
                """
                ## 📊 전략 로직

                ### 롱 진입 조건
                1. 캔들이 200일 EMA 위에 위치
                2. 스토캐스틱 RSI: %K선이 %D선을 골든크로스 + %D < 20
                3. 하이킨아시: 몸통이 이전보다 크고 + 아래꼬리 없음

                ### 숏 진입 조건
                1. 캔들이 200일 EMA 아래에 위치
                2. 스토캐스틱 RSI: %K선이 %D선을 데드크로스 + %D > 80
                3. 하이킨아시: 몸통이 이전보다 크고 + 윗꼬리 없음

                ### 청산 조건
                - **Stop Loss**: 진입가 대비 -1.5% (기본)
                - **Take Profit**: 진입가 대비 +2% (기본)
                - 손익비 1.5% : 2%
                """
            )
        )
        st.markdown(
            textwrap.dedent(
                """
                ## ⚙️ 설정 가능 파라미터

                | 파라미터 | 기본값 | 설명 |
                |---------|--------|------|
                | 캔들 타입 | 하이킨아시 | 일반 캔들로 변경 가능 |
                | 타임프레임 | 5분 | 5/10/15분 선택 |
                | 이동평균선 | EMA 200 | 기간 조정 가능 |
                | 레버리지 | 1배 | 1~30배 |
                | 손절 | 1.5% | 조정 가능 |
                | 익절 | 2% | 조정 가능 |
                """
            )
        )
        st.markdown(
            textwrap.dedent(
                """
                **옵션 활용 가이드**
                - **캔들 타입**: 하이킨아시 기본값은 노이즈를 완화해 추세를 쉽게 파악하게 돕고, 급격한 변동을 직접 확인하고 싶다면 일반 캔들로 바꿔보세요.
                - **타임프레임**: 기본 5분 봉이며 시간을 늘릴수록 신호는 줄어들고 큰 추세 확인에 유리합니다.
                - **이동평균선 (EMA 200)**: 장기 추세 기준선으로 기간을 줄이면 단기 변동에 더 빨리 반응합니다.
                - **레버리지 (기본 1배)**: 수익과 손실 폭을 동시에 키우므로 변동성이 큰 장세에서는 보수적인 수치를 유지하세요.
                - **손절 (-1.5%)**: 자본 보호 기준이며 시장 상황에 따라 손실 허용 범위를 조절할 수 있습니다.
                - **익절 (+2%)**: 목표 수익률 설정값으로 추세가 길게 이어질 것으로 보이면 한도를 높여 전략을 테스트해보세요.
                """
            )
        )
        return

    price_df = load_price_data(data_start, data_end, state.timeframe)
    if price_df.empty:
        st.error("Bybit API에서 가격 데이터를 불러오지 못했습니다. 네트워크 상태와 날짜 범위를 확인해주세요.")
        return

    strategy_params = StrategyParameters(
        ma_period=state.ma_period,
        rsi_period=state.rsi_period,
        stoch_period=state.stoch_period,
        stoch_k=state.stoch_k,
        stoch_d=state.stoch_d,
        cooldown_minutes=state.ma_cooldown_minutes,
    )
    strategy = MA200StochRSIStrategy(params=strategy_params)
    engine = BacktestEngine()
    engine.default_trade_capital = state.position_capital

    logger.info(
        "Backtest start | %s ~ %s | timeframe %s | leverage x%d | TP %.2f%% | SL %.2f%% | cooldown %d min",
        state.start_date.isoformat(),
        state.end_date.isoformat(),
        state.timeframe,
        state.leverage,
        state.take_profit_pct * 100,
        state.stop_loss_pct * 100,
        state.ma_cooldown_minutes,
    )

    report = engine.run(
        price_df,
        strategy=strategy,
        initial_capital=initial_capital,
        leverage=state.leverage,
        take_profit_pct=state.take_profit_pct,
        stop_loss_pct=state.stop_loss_pct,
    )

    trades_count = len(report.trades) if report.trades is not None else 0
    total_return = report.metrics.get("total_return") if report.metrics else None
    if isinstance(total_return, (int, float)) and not np.isnan(total_return):
        logger.info(
            "Backtest end   | trades %d | total return %.2f%%",
            trades_count,
            total_return * 100,
        )
    else:
        logger.info("Backtest end   | trades %d | total return N/A", trades_count)

    equity_curve = report.equity_curve

    context = build_context(state)
    price_frame = getattr(report, "price_frame", price_df)
    saved_id = storage.save_backtest_result(report, price_frame, context)
    logger.info("Backtest saved | record id %s", saved_id)
    st.success(f"백테스트 결과가 저장되었습니다. 기록 ID: {saved_id}")

    render_backtest_results(
        price_df=price_frame,
        equity_curve=equity_curve,
        metrics=report.metrics,
        trades_df=report.trades,
        initial_capital=initial_capital,
        position_capital=state.position_capital,
        take_profit_pct=state.take_profit_pct,
        stop_loss_pct=state.stop_loss_pct,
        ma_cooldown_minutes=state.ma_cooldown_minutes,
        context=context,
    )


if __name__ == "__main__":
    main()
