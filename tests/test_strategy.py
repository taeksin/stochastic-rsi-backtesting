from __future__ import annotations

import pandas as pd

from src.strategy.ma200_stochrsi import MA200StochRSIStrategy, StrategyParameters


def sample_price_data(size: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=size, freq="5T")
    price = pd.Series(range(size), dtype=float) + 100
    return pd.DataFrame({"timestamp": dates, "open": price, "high": price, "low": price, "close": price})


def test_strategy_generates_signal_column():
    strategy = MA200StochRSIStrategy()
    df = sample_price_data()
    result = strategy.generate_signals(df)
    assert "signal" in result.columns
    assert "position" in result.columns


def test_position_size_uses_leverage():
    strategy = MA200StochRSIStrategy()
    size = strategy.calculate_position_size(1000, leverage=5)
    assert size == 5000


def test_ma_cooldown_blocks_new_entries(monkeypatch):
    timestamps = pd.date_range("2024-01-01", periods=4, freq="30T")
    close = pd.Series([99.0, 101.0, 102.0, 103.0])
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    )

    def fake_sma(dataframe: pd.DataFrame, period: int) -> pd.Series:
        return pd.Series([100.0, 100.0, 100.0, 100.0], index=dataframe.index)

    def fake_stoch(dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rsi": [30.0, 30.0, 30.0, 30.0],
                "stoch_rsi": [10.0, 20.0, 30.0, 40.0],
                "stoch_k": [10.0, 25.0, 35.0, 45.0],
                "stoch_d": [30.0, 10.0, 15.0, 25.0],
            },
            index=dataframe.index,
        )

    def fake_detect(dataframe: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "golden_cross": [False, True, False, False],
                "dead_cross": [False, False, False, False],
            },
            index=dataframe.index,
        )

    monkeypatch.setattr("src.strategy.ma200_stochrsi.calculate_sma", fake_sma)
    monkeypatch.setattr("src.strategy.ma200_stochrsi.stochastic_rsi.calculate", fake_stoch)
    monkeypatch.setattr("src.strategy.ma200_stochrsi.stochastic_rsi.detect_crossover", fake_detect)

    strategy = MA200StochRSIStrategy(
        StrategyParameters(cooldown_minutes=60, oversold=50.0, overbought=50.0)
    )
    result = strategy.generate_signals(df)

    assert result.loc[1, "ma_bias_change"] is True
    assert result.loc[1, "cooldown_active"] is True
    assert result.loc[1, "signal"] == 0
