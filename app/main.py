from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
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
from src.strategy.ma200_stochrsi import MA200StochRSIStrategy, StrategyParameters
from src.utils.logger import configure_logging
from src.utils import storage


def generate_mock_data(
    start: datetime, end: datetime, timeframe: str, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, end=end, freq=timeframe)
    steps = rng.normal(loc=0, scale=0.002, size=len(index))
    price = 20000 + np.cumsum(steps) * 20000
    price = np.maximum(price, 1000)
    df = pd.DataFrame(
        {
            "timestamp": index,
            "open": price,
            "high": price * (1 + rng.uniform(0, 0.002, len(index))),
            "low": price * (1 - rng.uniform(0, 0.002, len(index))),
            "close": price * (1 + rng.normal(0, 0.001, len(index))),
            "volume": rng.uniform(10, 100, len(index)),
        }
    )
    return df


def format_trade_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "진입 시각",
        "청산 시각",
        "포지션",
        "수익률(%)",
        "손익 (₩)",
        "투입 자본 (₩)",
        "잔액 (₩)",
        "종료 사유",
    ]
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=columns)

    df = trades_df.copy()

    entry_time = (
        pd.to_datetime(df.get("entry_time"), errors="coerce")
        if "entry_time" in df
        else pd.Series(pd.NaT, index=df.index)
    )
    exit_time = (
        pd.to_datetime(df.get("exit_time"), errors="coerce")
        if "exit_time" in df
        else pd.Series(pd.NaT, index=df.index)
    )
    entry_time = entry_time.dt.tz_localize(None)
    exit_time = exit_time.dt.tz_localize(None)

    direction = df["direction"].copy() if "direction" in df else pd.Series("", index=df.index)
    direction = direction.map({"long": "롱", "short": "숏"}).fillna(direction)

    pnl_pct = pd.to_numeric(df.get("pnl_pct"), errors="coerce") if "pnl_pct" in df else pd.Series(0, index=df.index)
    pnl_pct = pnl_pct * 100
    pnl_value = pd.to_numeric(df.get("pnl_value"), errors="coerce") if "pnl_value" in df else pd.Series(0, index=df.index)
    capital_used = pd.to_numeric(df.get("trade_capital"), errors="coerce") if "trade_capital" in df else pd.Series(0, index=df.index)
    balance = pd.to_numeric(df.get("balance"), errors="coerce") if "balance" in df else pd.Series(0, index=df.index)
    reason = df.get("exit_reason") if "exit_reason" in df else pd.Series("", index=df.index)
    reason = reason.map(
        {
            "take_profit": "익절",
            "stop_loss": "손절",
            "strategy_exit": "전략 종료",
            "reverse": "포지션 전환",
            "end_of_data": "기간 종료",
        }
    ).fillna(reason)

    formatted = pd.DataFrame(
        {
            "진입 시각": entry_time.dt.strftime("%Y-%m-%d %H:%M"),
            "청산 시각": exit_time.dt.strftime("%Y-%m-%d %H:%M"),
            "포지션": direction,
            "수익률(%)": pnl_pct.map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
            "손익 (₩)": pnl_value.map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"),
            "투입 자본 (₩)": capital_used.map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"),
            "잔액 (₩)": balance.map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"),
            "종료 사유": reason,
        }
    )
    return formatted


def render_backtest_results(
    price_df: pd.DataFrame,
    equity_curve: pd.Series,
    metrics: Dict[str, Any],
    trades_df: pd.DataFrame,
    initial_capital: float,
    position_capital: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> None:
    charts.render_price_chart(price_df)
    charts.render_equity_curve(equity_curve)
    metrics_display.render_metrics(metrics)

    st.subheader("거래 내역")
    st.caption(
        f"초기 자본: {initial_capital:,.0f} | 거래당 투입 자본: {position_capital:,.0f} | "
        f"손절: {stop_loss_pct * 100:.2f}% | 익절: {take_profit_pct * 100:.2f}%"
    )

    display_df = format_trade_table(trades_df)
    if display_df.empty:
        st.info("거래 내역이 없습니다.")
    else:
        st.dataframe(display_df, use_container_width=True)


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
        },
    }


def load_saved_result(backtest_id: str) -> Optional[Dict[str, Any]]:
    data = storage.load_backtest(backtest_id)
    if not data:
        return None

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

    equity_curve = storage.load_equity_curve_csv(data)
    trades_df = storage.load_trades_csv(data)

    return {
        "price_df": price_df,
        "equity_curve": equity_curve,
        "metrics": data.get("metrics", {}),
        "trades": trades_df,
        "context": data.get("context", {}),
    }


def main() -> None:
    configure_logging()
    st.set_page_config(page_title="Stochastic RSI Backtester", layout="wide")
    st.title("📊 MA200 + Stochastic RSI 백테스팅")

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
        saved_results=saved_backtests,
    )

    if state.mode == "분석":
        if not state.selected_backtest_id:
            st.info("사이드바에서 분석할 테스트를 선택하세요.")
            return

        saved = load_saved_result(state.selected_backtest_id)
        if saved is None:
            st.error("선택한 테스트 기록을 불러올 수 없습니다.")
            return

        context = saved.get("context", {})
        parameters = context.get("parameters", {})
        settings_ctx = context.get("settings", {})
        tp_ctx = settings_ctx.get("take_profit_pct", default_take_profit_pct)
        sl_ctx = settings_ctx.get("stop_loss_pct", default_stop_loss_pct)

        st.subheader("저장된 테스트 결과")
        summary_lines = []
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
            f"TP {tp_ctx * 100:.2f}% / SL {sl_ctx * 100:.2f}%"
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
        )
        return

    data_start = datetime.combine(state.start_date, datetime.min.time())
    data_end = datetime.combine(state.end_date, datetime.min.time())

    price_df = generate_mock_data(data_start, data_end, state.timeframe)

    strategy_params = StrategyParameters(
        ma_period=state.ma_period,
        rsi_period=state.rsi_period,
        stoch_period=state.stoch_period,
        stoch_k=state.stoch_k,
        stoch_d=state.stoch_d,
    )
    strategy = MA200StochRSIStrategy(params=strategy_params)
    engine = BacktestEngine()
    engine.default_trade_capital = state.position_capital

    if not state.run_backtest:
        st.info("사이드바에서 파라미터를 설정한 뒤 '백테스트 실행' 버튼을 눌러주세요.")
        return

    report = engine.run(
        price_df,
        strategy=strategy,
        initial_capital=initial_capital,
        leverage=state.leverage,
        take_profit_pct=state.take_profit_pct,
        stop_loss_pct=state.stop_loss_pct,
    )

    equity_curve = pd.Series(
        report.equity_curve.values,
        index=pd.to_datetime(price_df["timestamp"]),
    )

    context = build_context(state)
    saved_id = storage.save_backtest_result(report, price_df, context)
    st.success(f"테스트 결과가 저장되었습니다. 기록 ID: {saved_id}")

    render_backtest_results(
        price_df=price_df,
        equity_curve=equity_curve,
        metrics=report.metrics,
        trades_df=report.trades,
        initial_capital=initial_capital,
        position_capital=state.position_capital,
        take_profit_pct=state.take_profit_pct,
        stop_loss_pct=state.stop_loss_pct,
    )


if __name__ == "__main__":
    main()
