from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config.settings import settings


@dataclass(slots=True)
class DatabaseConnector:
    engine: Optional[Engine] = None

    def connect(self) -> Engine:
        if self.engine is None:
            self.engine = create_engine(settings.database.sqlalchemy_uri)
        return self.engine

    @contextmanager
    def session(self):
        engine = self.connect()
        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    def fetch_klines(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[Iterable[str]] = None,
        table: str = "klines_1m",
    ) -> pd.DataFrame:
        query = [
            "SELECT timestamp, symbol, open, high, low, close, volume",
            f"FROM {table}",
            "WHERE timestamp BETWEEN :start AND :end",
        ]
        params = {"start": start_date, "end": end_date}
        if symbols:
            query.append("AND symbol = ANY(:symbols)")
            params["symbols"] = list(symbols)

        sql = " ".join(query) + " ORDER BY timestamp ASC"

        with self.session() as conn:
            df = pd.read_sql(text(sql), conn, params=params, parse_dates=["timestamp"])
        return df

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
