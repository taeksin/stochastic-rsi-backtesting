from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def sqlalchemy_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(slots=True)
class BacktestDefaults:
    timeframe: str
    initial_capital: float
    leverage: int
    position_capital: float
    take_profit_pct: float
    stop_loss_pct: float


@dataclass(slots=True)
class Settings:
    database: DatabaseSettings
    backtest: BacktestDefaults


def _env_path() -> Optional[Path]:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    return env_path if env_path.exists() else None


def load_settings() -> Settings:
    env_file = _env_path()
    if env_file:
        load_dotenv(env_file, override=False)

    database = DatabaseSettings(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        name=os.getenv("DB_NAME", "crypto"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )
    backtest = BacktestDefaults(
        timeframe=os.getenv("DEFAULT_TIMEFRAME", "5T"),
        initial_capital=float(os.getenv("DEFAULT_INITIAL_CAPITAL", "1000000")),
        leverage=int(os.getenv("DEFAULT_LEVERAGE", "10")),
        position_capital=float(os.getenv("DEFAULT_POSITION_CAPITAL", "1000000")),
        take_profit_pct=float(os.getenv("DEFAULT_TAKE_PROFIT", "0.02")),
        stop_loss_pct=float(os.getenv("DEFAULT_STOP_LOSS", "0.015")),
    )
    return Settings(database=database, backtest=backtest)


settings = load_settings()
