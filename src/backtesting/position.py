from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Position:
    side: str
    entry_price: float
    size: float
    quantity: float
    leverage: int
    opened_at: datetime

    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    def unrealized_pnl(self, price: float) -> float:
        price_diff = (price - self.entry_price) * self.direction()
        return price_diff * self.quantity * self.leverage
