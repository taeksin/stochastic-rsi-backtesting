from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
    ma_cooldown_minutes: int
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
    default_ma_cooldown: int,
    saved_results: List[Dict[str, Any]],
) -> SidebarState:
    st.sidebar.header("설정")
    mode = st.sidebar.radio("모드 선택", ("백테스트", "분석"))
    st.sidebar.caption("백테스트: 새 파라미터로 시뮬레이션 실행 | 분석: 저장된 기록 확인")
    st.sidebar.markdown(f"**초기 자본:** {_format_currency(initial_capital)}")
    default_position_capital = (
        min(default_position_capital, initial_capital) if initial_capital else default_position_capital
    )

    today = date(2024, 12, 31)
    default_start = date(2024, 1, 1)

    start_date = default_start
    end_date = today
    timeframe_options = ["5T", "10T", "15T", "30T", "60T", "120T"]
    timeframe = default_timeframe if default_timeframe in timeframe_options else "5T"
    leverage = default_leverage
    ma_period = 200
    rsi_period = 14
    stoch_period = 14
    stoch_k = 3
    stoch_d = 3
    position_capital = default_position_capital
    take_profit_pct = default_take_profit_pct
    stop_loss_pct = default_stop_loss_pct
    ma_cooldown_minutes = default_ma_cooldown
    run_backtest = False
    selected_backtest_id: Optional[str] = None

    if mode == "백테스트":
        st.sidebar.subheader("백테스트 범위")
        start_date = st.sidebar.date_input("시작일", value=default_start)
        st.sidebar.caption("시작일을 앞당기면 더 많은 과거 데이터를 포함하고, 늦추면 기간이 짧아집니다.")
        end_date = st.sidebar.date_input("종료일", value=today)
        st.sidebar.caption("종료일을 뒤로 미루면 최신 데이터까지 포함하지만 계산량이 늘어납니다.")

        timeframe = st.sidebar.selectbox(
            "캔들 간격",
            options=timeframe_options,
            index=timeframe_options.index(default_timeframe) if default_timeframe in timeframe_options else 0,
        )
        st.sidebar.caption("간격을 줄이면 신호가 민감해지고 계산량이 늘어나며, 늘리면 추세가 부드러워집니다.")

        leverage = st.sidebar.slider("레버리지", min_value=1, max_value=30, value=default_leverage)
        st.sidebar.caption("레버리지를 높이면 수익과 손실 폭이 모두 커집니다.")

        st.sidebar.subheader("지표 파라미터")
        ma_period = st.sidebar.slider("이동평균 기간 (MA)", min_value=50, max_value=300, value=200)
        st.sidebar.caption("기간을 늘리면 추세가 부드러워지고, 줄이면 최근 변동에 더 민감해집니다.")
        rsi_period = st.sidebar.slider("RSI 기간", min_value=5, max_value=30, value=14)
        st.sidebar.caption("RSI 기간을 늘리면 움직임이 완만해지고, 줄이면 빠르게 반응합니다.")
        stoch_period = st.sidebar.slider("Stochastic RSI 기간", min_value=5, max_value=30, value=14)
        st.sidebar.caption("기간이 길수록 스토캐스틱 RSI가 평활해지고, 짧을수록 신호가 잦아집니다.")
        stoch_k = st.sidebar.slider("Stoch K", min_value=1, max_value=10, value=3)
        st.sidebar.caption("K 값을 높이면 교차 신호가 느려지고, 낮추면 민감해집니다.")
        stoch_d = st.sidebar.slider("Stoch D", min_value=1, max_value=10, value=3)
        st.sidebar.caption("D 값을 높이면 노이즈가 줄지만 신호가 늦어지고, 낮추면 잦아집니다.")

        step_value = max(int(initial_capital // 20), 1)
        position_capital = st.sidebar.slider(
            "거래당 투입 자본 (₩)",
            min_value=1,
            max_value=int(initial_capital),
            value=int(default_position_capital),
            step=step_value,
        )
        st.sidebar.caption("투입 자본을 늘리면 거래당 위험과 잠재 수익이 함께 증가합니다.")

        st.sidebar.subheader("리스크 관리")
        take_profit_input = st.sidebar.slider(
            "익절 기준 (%)",
            min_value=1.01,
            max_value=5.0,
            value=default_take_profit_pct * 100,
            step=0.01,
        )
        st.sidebar.caption("익절 비율을 높이면 목표 수익이 커지지만 달성 확률이 낮아질 수 있습니다.")
        stop_loss_input = st.sidebar.slider(
            "손절 기준 (%)",
            min_value=1.01,
            max_value=5.0,
            value=default_stop_loss_pct * 100,
            step=0.01,
        )
        st.sidebar.caption("손절 비율을 낮추면 손실을 빠르게 제한하고, 높이면 더 넓은 변동을 허용합니다.")

        take_profit_pct = take_profit_input / 100
        stop_loss_pct = stop_loss_input / 100

        ma_cooldown_minutes = st.sidebar.slider(
            "MA 모드 전환 쿨다운 (분)",
            min_value=0,
            max_value=1440,
            value=default_ma_cooldown,
            step=60,
        )
        st.sidebar.caption("쿨다운을 늘리면 모드 전환 후 재진입까지 기다리는 시간이 길어집니다.")

        run_backtest = st.sidebar.button("백테스트 실행")
    else:
        st.sidebar.subheader("저장된 테스트 기록")
        if saved_results:
            options: Dict[str, str] = {}
            for entry in saved_results:
                created = entry.get("created_at", "") or ""
                params = entry.get("parameters", {})
                start = params.get("start_date", "")
                end = params.get("end_date", "")
                total_return = entry.get("metrics", {}).get("total_return")
                label = f"{created} | {start}~{end}"
                if total_return is not None and not (isinstance(total_return, float) and math.isnan(total_return)):
                    label += f" | 수익률 {total_return * 100:.2f}%"
                options[label] = entry["id"]
            selected_label = st.sidebar.selectbox("테스트 선택", list(options.keys()))
            selected_backtest_id = options[selected_label]
        else:
            st.sidebar.info("저장된 백테스트가 없습니다.")
        take_profit_pct = default_take_profit_pct
        stop_loss_pct = default_stop_loss_pct
        ma_cooldown_minutes = default_ma_cooldown

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
        ma_cooldown_minutes=int(ma_cooldown_minutes),
        run_backtest=run_backtest,
        selected_backtest_id=selected_backtest_id,
    )
