from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MAX_CANDLE_POINTS = 1500
MAX_EQUITY_POINTS = 2000


def _downsample_ohlc(df: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, bool]:
    if df.empty or len(df) <= limit:
        return df, False
    frame = df.sort_values("timestamp").reset_index(drop=True)
    step = int(np.ceil(len(frame) / limit))
    groups = frame.index // step
    aggregated = (
        frame.assign(_group=groups)
        .groupby("_group", as_index=False)
        .agg(
            {
                "timestamp": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )
    return aggregated, True


def _downsample_series(series: pd.Series, limit: int) -> tuple[pd.Series, bool]:
    if series.empty or len(series) <= limit:
        return series, False
    step = int(np.ceil(len(series) / limit))
    indices = np.arange(0, len(series), step)
    if indices[-1] != len(series) - 1:
        indices = np.append(indices, len(series) - 1)
    sampled = series.iloc[indices]
    return sampled, True


def render_price_chart(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("시세 데이터가 비어 있어 차트를 표시할 수 없습니다.")
        return
    reduced_df, truncated = _downsample_ohlc(df, MAX_CANDLE_POINTS)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=reduced_df["timestamp"],
                open=reduced_df["open"],
                high=reduced_df["high"],
                low=reduced_df["low"],
                close=reduced_df["close"],
                name="Price",
            )
        ]
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
    if truncated:
        st.caption(f"차트는 약 {len(reduced_df)}개의 캔들로 축약해 표시합니다.")


def render_equity_curve(equity_curve: pd.Series) -> None:
    if equity_curve.empty:
        st.warning("백테스트를 실행하면 자산 곡선을 표시합니다.")
        return
    reduced_series, truncated = _downsample_series(equity_curve.sort_index(), MAX_EQUITY_POINTS)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=reduced_series.index,
                y=reduced_series.values,
                mode="lines",
                name="Equity",
            )
        ]
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
    if truncated:
        st.caption(f"자산 곡선은 약 {len(reduced_series)}개의 포인트로 축약해 표시합니다.")
