from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_price_chart(df):
    if df.empty:
        st.warning("표시할 가격 데이터가 없습니다.")
        return
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            )
        ]
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)


def render_equity_curve(equity_curve):
    if equity_curve.empty:
        st.warning("백테스트를 실행하면 자산 곡선이 표시됩니다.")
        return
    fig = go.Figure(
        data=[
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Equity",
            )
        ]
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
