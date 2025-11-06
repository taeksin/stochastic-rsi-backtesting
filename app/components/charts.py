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

    if "close" in df.columns and "timestamp" in df.columns and ("signal" in df.columns or "position" in df.columns):
        frame = df.sort_values("timestamp").reset_index(drop=True)
        signals = frame["signal"].fillna(0).astype(int) if "signal" in frame else frame["position"].fillna(0).diff().fillna(0).astype(int)
        entries = frame[signals > 0]
        exits = frame[signals < 0]

        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["timestamp"],
                    y=entries["close"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=9, color="#2ca02c"),
                    name="롱 진입",
                )
            )
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits["timestamp"],
                    y=exits["close"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=9, color="#d62728"),
                    name="숏 진입",
                )
            )
        if "exit_signal" in frame.columns:
            close_events = frame[frame["exit_signal"].astype(str) != ""]
            if not close_events.empty:
                fig.add_trace(
                    go.Scatter(
                        x=close_events["timestamp"],
                        y=close_events["close"],
                        mode="markers",
                        marker=dict(symbol="x", size=9, color="#ff7f0e"),
                        name="포지션 청산",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=reduced_df["timestamp"],
                y=reduced_df["close"],
                mode="lines",
                line=dict(color="#1f77b4", width=1.2),
                name="종가",
            )
        )
        if {"engine_exit_type", "engine_exit_price"}.issubset(frame.columns):
            engine_events = frame[pd.notna(frame["engine_exit_price"])]
            if not engine_events.empty:
                tp_events = engine_events[engine_events["engine_exit_type"] == "take_profit"]
                sl_events = engine_events[engine_events["engine_exit_type"] == "stop_loss"]
                if not tp_events.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=tp_events["timestamp"],
                            y=tp_events["engine_exit_price"],
                            mode="markers",
                            marker=dict(symbol="circle", size=8, color="#17becf"),
                            name="익절 청산",
                        )
                    )
                if not sl_events.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sl_events["timestamp"],
                            y=sl_events["engine_exit_price"],
                            mode="markers",
                            marker=dict(symbol="x", size=8, color="#d62728"),
                            name="손절 청산",
                        )
                    )

    fig.update_layout(
        title="가격 차트",
        height=400,
        margin=dict(l=10, r=10, t=50, b=30),
    )
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
    fig.update_layout(title="자산 곡선", height=300, margin=dict(l=10, r=10, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)
    if truncated:
        st.caption(f"자산 곡선은 약 {len(reduced_series)}개의 포인트로 축약해 표시합니다.")
