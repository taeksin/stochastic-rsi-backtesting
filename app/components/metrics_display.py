from __future__ import annotations

import math
from typing import Dict

import streamlit as st


def _format_percent(value: float) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_currency(value: float) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(number):
        return "N/A"
    return f"{number:,.0f}"


def _format_number(value: float) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(number):
        return "N/A"
    return f"{number:.2f}"


def render_metrics(summary: Dict[str, float]) -> None:
    if not summary:
        st.info("백테스트를 실행하면 성과 지표가 표시됩니다.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("총 수익률", _format_percent(summary.get("total_return", float("nan"))))
    col2.metric(
        "샤프 비율",
        _format_number(summary.get("sharpe_ratio")),
    )
    col3.metric(
        "최대 낙폭 (MDD)",
        _format_percent(summary.get("max_drawdown", float("nan"))),
    )

    col4, col5 = st.columns(2)
    col4.metric(
        "승률",
        _format_percent(summary.get("win_rate", float("nan"))),
    )
    col5.metric(
        "총 거래 수",
        f"{summary.get('trades', 0)}",
    )

    col6, col7 = st.columns(2)
    initial_capital = summary.get("initial_capital")
    final_equity = summary.get("final_equity")
    profit_value = None
    try:
        if final_equity is not None and initial_capital is not None:
            profit_value = float(final_equity) - float(initial_capital)
    except (TypeError, ValueError):
        profit_value = None

    col6.metric(
        "거래당 투입자본",
        _format_currency(summary.get("trade_capital", float("nan"))),
    )
    col7.metric(
        "최종 잔액",
        _format_currency(summary.get("final_equity", float("nan"))),
    )

    col8, _ = st.columns(2)
    col8.metric(
        "총 손익",
        _format_currency(profit_value),
    )
