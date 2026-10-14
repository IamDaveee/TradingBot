from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading_bot.backtesting.models import (
    OrderRequest,
)
from trading_bot.backtesting.portfolio import (
    Position,
)
from trading_bot.market_data.candle import (
    TickerCandle,
)


@dataclass(frozen=True, slots=True)
class StrategyContext:
    cash: Decimal
    equity: Decimal
    position: Position


class Strategy(Protocol):
    def on_candle(
        self,
        candle: TickerCandle,
        context: StrategyContext,
    ) -> list[OrderRequest]:
        """Process one closed candle and return new orders."""
        ...
