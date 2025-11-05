from __future__ import annotations

from unittest.mock import MagicMock

from src.data.database import DatabaseConnector


def test_connect_uses_sqlalchemy_engine(monkeypatch):
    fake_engine = MagicMock()

    def fake_create_engine(uri):
        fake_engine.uri = uri
        return fake_engine

    monkeypatch.setattr("src.data.database.create_engine", fake_create_engine)

    connector = DatabaseConnector()
    engine = connector.connect()

    assert engine is fake_engine
    assert connector.engine is fake_engine
