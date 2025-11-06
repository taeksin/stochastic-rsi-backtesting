from datetime import datetime
from src.data.bybit_client import fetch_bybit_klines

print("start")
data = fetch_bybit_klines(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 1, 0, 10),
    interval_minutes="1",
    symbol="BTCUSDT",
)
print(len(data))
print(data.head())
