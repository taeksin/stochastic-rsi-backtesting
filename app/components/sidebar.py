from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Any, Dict, List, Optional

import streamlit as st


@dataclass(slots=True)
class SidebarState:
    mode: str
    start_date: date
    end_date: date
    timeframe: str
    leverage: int
    ma_period: int
    rsi_period: int
    stoch_period: int
    stoch_k: int
    stoch_d: int
    position_capital: float
    take_profit_pct: float
    stop_loss_pct: float
    run_backtest: bool
    selected_backtest_id: Optional[str]


def _format_currency(value: float) -> str:
    return f"{value:,.0f} ₩"


def render_sidebar(
    default_timeframe: str,
    default_leverage: int,
    initial_capital: float,
    default_position_capital: float,
    default_take_profit_pct: float,
    default_stop_loss_pct: float,
    saved_results: List[Dict[str, Any]],
) -> SidebarState:
    st.sidebar.header("설정")
    mode = st.sidebar.radio("모드 선택", ("백테스트", "분석"))
    st.sidebar.markdown(f"**초기 자본:** {_format_currency(initial_capital)}")
    default_position_capital = (
        min(default_position_capital, initial_capital) if initial_capital else default_position_capital
    )

    today = date(2024, 12, 31)
    default_start = date(2024, 1, 1)

    start_date = default_start
    end_date = today
    timeframe = default_timeframe if default_timeframe in ["5T", "10T", "15T"] else "5T"
    leverage = default_leverage
    ma_period = 200
    rsi_period = 14
    stoch_period = 14
    stoch_k = 3
    stoch_d = 3
    position_capital = default_position_capital
    run_backtest = False
    selected_backtest_id: Optional[str] = None

    if mode == "백테스트":
        st.sidebar.subheader("백테스트 범위")
        start_date = st.sidebar.date_input("시작일", value=default_start)
        end_date = st.sidebar.date_input("종료일", value=today)

        timeframe = st.sidebar.selectbox(
            "타임프레임",
            options=["5T", "10T", "15T"],
            index=["5T", "10T", "15T"].index(default_timeframe)
            if default_timeframe in ["5T", "10T", "15T"]
            else 0,
        )

        leverage = st.sidebar.slider("레버리지", min_value=1, max_value=30, value=default_leverage)

        st.sidebar.subheader("지표 파라미터")
        ma_period = st.sidebar.slider("200일 이동평균 기간", min_value=50, max_value=300, value=200)
        rsi_period = st.sidebar.slider("RSI 기간", min_value=5, max_value=30, value=14)
        stoch_period = st.sidebar.slider("Stochastic 기간", min_value=5, max_value=30, value=14)
        stoch_k = st.sidebar.slider("Stoch K", min_value=1, max_value=10, value=3)
        stoch_d = st.sidebar.slider("Stoch D", min_value=1, max_value=10, value=3)

        step_value = max(int(initial_capital // 20), 1)
        position_capital = st.sidebar.slider(
            "거래당 투입 자본 (₩)",
            min_value=1,
            max_value=int(initial_capital),
            value=int(default_position_capital),
            step=step_value,
        )

        st.sidebar.subheader("리스크 관리")
        take_profit_input = st.sidebar.slider(
            "익절 기준 (%)",
            min_value=1.01,
            max_value=5.0,
            value=default_take_profit_pct * 100,
            step=0.01,
        )
        stop_loss_input = st.sidebar.slider(
            "손절 기준 (%)",
            min_value=1.01,
            max_value=5.0,
            value=default_stop_loss_pct * 100,
            step=0.01,
        )

        take_profit_pct = take_profit_input / 100
        stop_loss_pct = stop_loss_input / 100

        run_backtest = st.sidebar.button("백테스트 실행")
    else:
        st.sidebar.subheader("저장된 테스트 기록")
        if saved_results:
            options = {}
            for entry in saved_results:
                created = entry.get("created_at", "") or ""
                params = entry.get("parameters", {})
                start = params.get("start_date", "")
                end = params.get("end_date", "")
                total_return = entry.get("metrics", {}).get("total_return")
                label = f"{created} | {start}~{end}"
                if total_return is not None and not (
                    isinstance(total_return, float) and math.isnan(total_return)
                ):
                    label += f" | 수익률 {total_return*100:.2f}%"
                options[label] = entry["id"]
            selected_label = st.sidebar.selectbox("테스트 선택", list(options.keys()))
            selected_backtest_id = options[selected_label]
        else:
            st.sidebar.info("저장된 백테스트가 없습니다.")
        take_profit_pct = default_take_profit_pct
        stop_loss_pct = default_stop_loss_pct

    return SidebarState(
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        leverage=leverage,
        ma_period=ma_period,
        rsi_period=rsi_period,
        stoch_period=stoch_period,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        position_capital=float(position_capital),
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        run_backtest=run_backtest,
        selected_backtest_id=selected_backtest_id,
    )
